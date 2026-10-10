# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Torch reference/fixtures for tests and benchmarks, never production fallback."""

import torch


def sparse_paged_reference(
    q,
    k_cache,
    v_cache,
    topk_idx,
    block_table,
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    max_query_len,
    sm_scale,
    output=None,
    *,
    page_size=128,
    q_scale=None,
    k_scale=None,
    v_scale=None,
):
    """Vectorized reference: all tensor math remains on the input device."""
    del max_query_len
    if q.shape[0] == 0:
        return torch.empty_like(q)
    t, h, d = q.shape
    hk, _, topk = topk_idx.shape
    row = torch.arange(t, device=q.device)
    req = torch.bucketize(row, cu_seqlens_q[1:].contiguous(), right=True)
    qabs = prefix_lens[req] + row - cu_seqlens_q[req]
    pos = topk_idx.long().unsqueeze(-1) * 128 + torch.arange(128, device=q.device)
    valid = (topk_idx.unsqueeze(-1) >= 0) & (pos < seq_lens[req][None, :, None, None])
    valid &= pos <= qabs[None, :, None, None]
    slot = torch.arange(topk, device=q.device)[None, None, :, None]
    valid &= (
        slot
        < torch.minimum(torch.full_like(qabs, topk), (qabs + 128) // 128)[
            None, :, None, None
        ]
    )
    page_col = pos.div(page_size, rounding_mode="floor")
    valid &= (page_col >= 0) & (page_col < block_table.shape[1])
    page = block_table[
        req[None, :, None, None], page_col.clamp(0, block_table.shape[1] - 1)
    ]
    valid &= (page >= 0) & (page < k_cache.shape[0])
    page = page.clamp(0, k_cache.shape[0] - 1)
    heads = torch.arange(hk, device=q.device)[:, None, None, None]
    k = k_cache[page, pos.remainder(page_size), heads]
    v = v_cache[page, pos.remainder(page_size), heads]
    k = (k.float() * (1.0 if k_scale is None else k_scale)).to(q.dtype).float()
    v = (v.float() * (1.0 if v_scale is None else v_scale)).to(q.dtype).float()
    k = k.reshape(hk, t, topk * 128, d)
    v = v.reshape(hk, t, topk * 128, d)
    scores = torch.einsum("thgd,htnd->htgn", q.float().reshape(t, hk, h // hk, d), k)
    scores *= sm_scale * (1.0 if q_scale is None else q_scale)
    scores = scores.masked_fill(~valid.reshape(hk, t, 1, topk * 128), float("-inf"))
    prob = torch.softmax(scores, -1).nan_to_num(0)
    result = torch.einsum("htgn,htnd->thgd", prob, v).reshape(t, h, d).to(q.dtype)
    if output is not None:
        output.copy_(result)
        return output
    return result


def make_sparse_paged_inputs(
    batch, qlen, context, page_size, dtype, device="cuda", kv_dtype=None
):
    """Small deterministic inputs; metadata preparation stays outside timing."""
    if context < qlen:
        raise ValueError("context must include all query tokens")
    generator = torch.Generator(device=device).manual_seed(20261010)
    pages_per_req = (context + page_size - 1) // page_size
    total_q = batch * qlen
    q = torch.randn((total_q, 8, 128), generator=generator, dtype=dtype, device=device)
    shape = (batch * pages_per_req, page_size, 1, 128)
    k = torch.randn(shape, generator=generator, dtype=dtype, device=device)
    v = torch.randn(shape, generator=generator, dtype=dtype, device=device)
    if kv_dtype is not None:
        k, v = k.to(kv_dtype), v.to(kv_dtype)
    topk = (
        torch.arange(16, dtype=torch.int32, device=device)
        .view(1, 1, 16)
        .repeat(1, total_q, 1)
    )
    topk[:, :, 0] = (context - 1) // 128
    return dict(
        q=q,
        k_cache=k,
        v_cache=v,
        topk_idx=topk,
        block_table=torch.arange(
            batch * pages_per_req, dtype=torch.int32, device=device
        ).view(batch, pages_per_req),
        cu_seqlens_q=torch.arange(batch + 1, dtype=torch.int32, device=device) * qlen,
        seq_lens=torch.full((batch,), context, dtype=torch.int32, device=device),
        prefix_lens=torch.full(
            (batch,), context - qlen, dtype=torch.int32, device=device
        ),
        max_query_len=qlen,
        sm_scale=128**-0.5,
        output=torch.empty_like(q),
        page_size=page_size,
    )
