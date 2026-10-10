# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only host-contract checks; launch recorders never execute a kernel."""

import importlib.util
import itertools
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


class TensorMetadata:
    _ids = itertools.count()

    def __init__(
        self, shape, dtype=torch.bfloat16, device=None, strides=None, storage=None
    ):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = device or torch.device("cuda:0")
        self.requires_grad = False
        self.storage = next(self._ids) if storage is None else storage
        if strides is None:
            strides = []
            stride = 1
            for size in reversed(shape):
                strides.insert(0, stride)
                stride *= max(1, size)
        self.strides = tuple(strides)

    def stride(self, dim=None):
        return self.strides if dim is None else self.strides[dim]

    def numel(self):
        return math.prod(self.shape)

    def unsqueeze(self, dim):
        assert dim == 0
        return TensorMetadata(
            (1,) + self.shape,
            self.dtype,
            self.device,
            (self.numel(),) + self.strides,
            self.storage,
        )


class LaunchRecorder:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


@pytest.fixture
def op():
    source = (
        Path(__file__).parents[2] / "src/flag_attn/minimax_sparse_attention/paged_kv.py"
    )
    spec = importlib.util.spec_from_file_location(
        "sparse_paged_contract_target", source
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.torch = SimpleNamespace(
        Tensor=TensorMetadata,
        float16=torch.float16,
        bfloat16=torch.bfloat16,
        float32=torch.float32,
        int32=torch.int32,
        empty=lambda shape, dtype, device: TensorMetadata(shape, dtype, device),
        _C=SimpleNamespace(_overlaps=lambda x, y: x.storage == y.storage),
    )
    module._paged_sparse_kernel = LaunchRecorder()
    module._merge_paged_sparse_kernel = LaunchRecorder()
    return module


def inputs(t=4, batch=4, topk=16, page_size=128):
    return dict(
        q=TensorMetadata((t, 8, 128)),
        k_cache=TensorMetadata((16, page_size, 1, 128)),
        v_cache=TensorMetadata((16, page_size, 1, 128)),
        topk_idx=TensorMetadata((1, t, topk), torch.int32),
        block_table=TensorMetadata((batch, 16), torch.int32),
        seq_lens=TensorMetadata((batch,), torch.int32),
        output=TensorMetadata((t, 8, 128)),
        sm_scale=128**-0.5,
        page_size=page_size,
    )


@pytest.mark.parametrize("page_size", [1, 16, 128])
def test_prefill_launch_has_separate_cache_pointers(op, page_size):
    args = inputs(page_size=page_size)
    cu = TensorMetadata((5,), torch.int32)
    prefix = TensorMetadata((4,), torch.int32)
    result = op.minimax_m3_sparse_attn_paged(
        **args, cu_seqlens_q=cu, prefix_lens=prefix, max_query_len=1
    )
    grid, buffers, meta = op._paged_sparse_kernel.calls[0]
    assert result is args["output"]
    assert grid == (1, 4, 1)
    assert buffers[1] is args["k_cache"] and buffers[2] is args["v_cache"]
    assert meta["PAGE_SIZE"] == page_size
    assert not op._merge_paged_sparse_kernel.calls


@pytest.mark.parametrize(
    "t,qlen,topk,splits", [(4, 1, 16, 16), (8, 2, 13, 8), (256, 1, 16, 1)]
)
def test_decode_workspace_and_shape_only_splits(op, t, qlen, topk, splits):
    args = inputs(t=t, batch=t // qlen, topk=topk)
    scratch = op.minimax_m3_sparse_decode_workspace(args["q"], 1, topk)
    result = op.minimax_m3_sparse_attn_decode_paged(
        **args, decode_query_len=qlen, workspace=scratch
    )
    grid, buffers, meta = op._paged_sparse_kernel.calls[0]
    assert result is args["output"]
    assert grid == (t, 1, splits)
    assert meta["DECODE"] and meta["QLEN"] == qlen
    assert scratch[0].shape == (splits, t, 8, 128)
    assert scratch[0].dtype == torch.float32
    if splits > 1:
        assert buffers[8] is scratch[0] and buffers[9] is scratch[1]
        assert op._merge_paged_sparse_kernel.calls[0][0] == (t, 8)
    else:
        assert not op._merge_paged_sparse_kernel.calls


def test_empty_has_no_launch(op):
    args = inputs(t=0, batch=0)
    assert op.minimax_m3_sparse_attn_decode_paged(**args) is args["output"]
    assert not op._paged_sparse_kernel.calls


def test_scale_tensor_is_forwarded_without_readback(op):
    args = inputs()
    scale = TensorMetadata((), torch.float32)
    op.minimax_m3_sparse_attn_decode_paged(**args, k_scale=scale, v_scale=0.5)
    _, buffers, meta = op._paged_sparse_kernel.calls[0]
    assert buffers[11] is scale
    assert meta["K_SCALE_PTR"] and not meta["V_SCALE_PTR"]


@pytest.mark.parametrize("scale", [0, -1, float("inf"), float("nan"), True])
def test_invalid_scale_fails_before_launch(op, scale):
    with pytest.raises(ValueError if not isinstance(scale, bool) else TypeError):
        op.minimax_m3_sparse_attn_decode_paged(**inputs(), k_scale=scale)
    assert not op._paged_sparse_kernel.calls


@pytest.mark.parametrize(
    "change,error",
    [
        ("cpu", NotImplementedError),
        ("dtype", NotImplementedError),
        ("broadcast", NotImplementedError),
        ("overlap", NotImplementedError),
        ("grad", NotImplementedError),
        ("alias", ValueError),
        ("workspace", ValueError),
        ("decode_length", ValueError),
    ],
)
def test_invalid_contract_fails_before_launch(op, change, error):
    args = inputs()
    extras = {}
    if change == "cpu":
        args["q"].device = torch.device("cpu")
    elif change == "dtype":
        args["q"].dtype = torch.float32
    elif change == "broadcast":
        args["q"].strides = (0, 128, 1)
    elif change == "overlap":
        args["q"].strides = (1, 128, 1)
    elif change == "grad":
        args["q"].requires_grad = True
    elif change == "alias":
        args["output"].storage = args["q"].storage
    elif change == "workspace":
        extras["workspace"] = (TensorMetadata((1,)), TensorMetadata((1,)))
    elif change == "decode_length":
        extras["decode_query_len"] = 2
    with pytest.raises(error):
        op.minimax_m3_sparse_attn_decode_paged(**args, **extras)
    assert not op._paged_sparse_kernel.calls


def test_noncontiguous_request_metadata_rejected(op):
    args = inputs()
    args["seq_lens"].strides = (2,)
    with pytest.raises(NotImplementedError, match="contiguous"):
        op.minimax_m3_sparse_attn_decode_paged(**args)


@pytest.mark.parametrize("page_size", [1, 16, 128])
def test_torch_reference_matches_simple_cpu_attention(page_size):
    source = Path(__file__).parents[2] / "src/flag_attn/testing/minimax_sparse_paged.py"
    spec = importlib.util.spec_from_file_location("sparse_paged_cpu_reference", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    a = module.make_sparse_paged_inputs(
        2, 3, 257, page_size, torch.bfloat16, device="cpu"
    )
    result = module.sparse_paged_reference(**{**a, "output": None})
    expected = torch.empty_like(result)
    for row in range(6):
        req = row // 3
        qabs = 254 + row % 3
        selected = []
        for block in a["topk_idx"][0, row, : min(16, (qabs + 128) // 128)].tolist():
            selected.extend(
                range(max(0, block * 128), min((block + 1) * 128, qabs + 1))
            )
        slots = torch.tensor(selected)
        page = a["block_table"][req, slots // page_size]
        k = a["k_cache"][page, slots % page_size, 0].float()
        v = a["v_cache"][page, slots % page_size, 0].float()
        logits = a["q"][row].float() @ k.T * a["sm_scale"]
        expected[row] = (torch.softmax(logits, -1) @ v).to(result.dtype)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
