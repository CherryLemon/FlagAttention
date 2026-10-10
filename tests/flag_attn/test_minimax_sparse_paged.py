# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Numerical and Graph tests, authored but deferred during development."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.getenv("FLAGGEMS_RUN_SPARSE_GPU_TESTS") != "1",
    reason="GPU regression explicitly deferred during development",
)


@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("page_size", [1, 16, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("qlen,context", [(1, 1), (3, 257), (5, 4096)])
def test_numerical(decode, page_size, dtype, qlen, context):
    from flag_attn import (
        minimax_m3_sparse_attn_decode_paged,
        minimax_m3_sparse_attn_paged,
    )
    from flag_attn.testing.minimax_sparse_paged import (
        make_sparse_paged_inputs,
        sparse_paged_reference,
    )

    a = make_sparse_paged_inputs(2, qlen, context, page_size, dtype)
    reference = sparse_paged_reference(**{**a, "output": None})
    if decode:
        result = minimax_m3_sparse_attn_decode_paged(
            a["q"],
            a["k_cache"],
            a["v_cache"],
            a["topk_idx"],
            a["block_table"],
            a["seq_lens"],
            a["sm_scale"],
            a["output"],
            qlen,
            page_size=page_size,
        )
    else:
        result = minimax_m3_sparse_attn_paged(**a)
    assert result is a["output"]
    torch.testing.assert_close(result, reference, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_fp8_scale_tensors(kv_dtype):
    from flag_attn import minimax_m3_sparse_attn_decode_paged
    from flag_attn.testing.minimax_sparse_paged import (
        make_sparse_paged_inputs,
        sparse_paged_reference,
    )

    a = make_sparse_paged_inputs(2, 2, 513, 16, torch.bfloat16, kv_dtype=kv_dtype)
    scales = dict(
        k_scale=torch.tensor(0.5, device="cuda"),
        v_scale=torch.tensor(1.5, device="cuda"),
    )
    reference = sparse_paged_reference(**{**a, "output": None}, **scales)
    result = minimax_m3_sparse_attn_decode_paged(
        a["q"],
        a["k_cache"],
        a["v_cache"],
        a["topk_idx"],
        a["block_table"],
        a["seq_lens"],
        a["sm_scale"],
        a["output"],
        2,
        page_size=16,
        **scales,
    )
    torch.testing.assert_close(result, reference, rtol=0.02, atol=0.02)


def test_empty_selected_blocks_zero_output():
    from flag_attn import minimax_m3_sparse_attn_decode_paged
    from flag_attn.testing.minimax_sparse_paged import make_sparse_paged_inputs

    a = make_sparse_paged_inputs(2, 1, 257, 128, torch.bfloat16)
    a["topk_idx"].fill_(-1)
    a["output"].fill_(float("nan"))
    minimax_m3_sparse_attn_decode_paged(
        a["q"],
        a["k_cache"],
        a["v_cache"],
        a["topk_idx"],
        a["block_table"],
        a["seq_lens"],
        a["sm_scale"],
        a["output"],
    )
    torch.testing.assert_close(
        a["output"], torch.zeros_like(a["output"]), rtol=0, atol=0
    )


@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_separate_cache_matches_existing_packed_interface(decode, kv_dtype):
    from flag_attn import (
        minimax_m3_sparse_attn,
        minimax_m3_sparse_attn_decode,
        minimax_m3_sparse_attn_decode_paged,
        minimax_m3_sparse_attn_paged,
    )
    from flag_attn.testing.minimax_sparse_paged import make_sparse_paged_inputs

    a = make_sparse_paged_inputs(2, 2, 513, 128, torch.bfloat16, kv_dtype=kv_dtype)
    # Packing is test-only; production adapters preserve the original storage.
    packed = (
        torch.cat((a["k_cache"], a["v_cache"]), -1).permute(0, 2, 1, 3).contiguous()
    )
    expected = torch.empty_like(a["q"])
    scales = (
        dict(
            k_scale=torch.tensor(0.5, device="cuda"),
            v_scale=torch.tensor(1.5, device="cuda"),
        )
        if kv_dtype == torch.float8_e4m3fn
        else {}
    )
    if decode:
        minimax_m3_sparse_attn_decode(
            a["q"],
            packed,
            a["topk_idx"],
            a["block_table"],
            a["seq_lens"],
            1,
            a["sm_scale"],
            expected,
            2,
            **scales,
        )
        result = minimax_m3_sparse_attn_decode_paged(
            a["q"],
            a["k_cache"],
            a["v_cache"],
            a["topk_idx"],
            a["block_table"],
            a["seq_lens"],
            a["sm_scale"],
            a["output"],
            2,
            **scales,
        )
    else:
        minimax_m3_sparse_attn(
            a["q"],
            packed,
            a["topk_idx"],
            a["block_table"],
            a["cu_seqlens_q"],
            a["seq_lens"],
            a["prefix_lens"],
            a["max_query_len"],
            1,
            a["sm_scale"],
            expected,
            **scales,
        )
        result = minimax_m3_sparse_attn_paged(**a, **scales)
    torch.testing.assert_close(result, expected, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("decode", [False, True])
def test_graph_replay_refreshes_metadata(decode):
    from flag_attn import (
        minimax_m3_sparse_attn_decode_paged,
        minimax_m3_sparse_attn_paged,
        minimax_m3_sparse_decode_workspace,
    )
    from flag_attn.testing.minimax_sparse_paged import (
        make_sparse_paged_inputs,
        sparse_paged_reference,
    )

    a = make_sparse_paged_inputs(2, 2, 513, 16, torch.bfloat16)
    scratch = minimax_m3_sparse_decode_workspace(a["q"], 1, 16)

    def run():
        if decode:
            return minimax_m3_sparse_attn_decode_paged(
                a["q"],
                a["k_cache"],
                a["v_cache"],
                a["topk_idx"],
                a["block_table"],
                a["seq_lens"],
                a["sm_scale"],
                a["output"],
                2,
                page_size=16,
                workspace=scratch,
            )
        return minimax_m3_sparse_attn_paged(**a)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        run()
    # Update device buffers without replacing their storage.
    a["seq_lens"].sub_(128)
    a["prefix_lens"].sub_(128)
    a["q"].mul_(0.5)
    graph.replay()
    reference = sparse_paged_reference(**{**a, "output": None})
    torch.testing.assert_close(a["output"], reference, rtol=0.02, atol=0.02)
