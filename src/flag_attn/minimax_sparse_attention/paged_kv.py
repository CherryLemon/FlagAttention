# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Causal M3 GQA over separate paged K/V buffers.

Logical sparse blocks contain 128 tokens. Physical KV pages may have a
different size. Callers provide logical block IDs, physical page tables and
device sequence metadata; no cache packing or framework imports are needed.
This initial path uses conservative fixed launch settings, pending GPU tuning.
"""

import math

import torch
import triton
import triton.language as tl

__all__ = [
    "minimax_m3_sparse_attn_paged",
    "minimax_m3_sparse_attn_decode_paged",
    "minimax_m3_sparse_decode_workspace",
]

_FLOAT8 = tuple(
    getattr(torch, name)
    for name in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz")
    if hasattr(torch, name)
)


@triton.jit
def _paged_sparse_kernel(
    Q,
    K,
    V,
    TopK,
    Pages,
    CuQ,
    Seq,
    Prefix,
    Out,
    Lse,
    QScale,
    KScale,
    VScale,
    T: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_PAGES: tl.constexpr,
    PAGE_COLS: tl.constexpr,
    TOPK: tl.constexpr,
    DECODE: tl.constexpr,
    QLEN: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    SCALE: tl.constexpr,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    SV: tl.constexpr,
    ST: tl.constexpr,
    SP: tl.constexpr,
    SO: tl.constexpr,
    SL: tl.constexpr,
    Q_SCALE_PTR: tl.constexpr,
    K_SCALE_PTR: tl.constexpr,
    V_SCALE_PTR: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    head_req = tl.program_id(1)
    split = tl.program_id(2)
    if DECODE:
        req = row // QLEN
        local_q = row % QLEN
        kh = head_req
        qidx = row
        qlen = QLEN
        seq_len = tl.load(Seq + req)
        qabs = seq_len - QLEN + local_q
    else:
        req = head_req // HK
        kh = head_req % HK
        start = tl.load(CuQ + req)
        qlen = tl.load(CuQ + req + 1) - start
        local_q = row
        qidx = start + row
        seq_len = tl.load(Seq + req)
        qabs = tl.load(Prefix + req) + local_q
    if local_q >= qlen or qidx >= T:
        return

    gh = HQ // HK
    h = kh * gh + tl.arange(0, BLOCK_H)
    d = tl.arange(0, BLOCK_D)
    n = tl.arange(0, 128)
    q = tl.load(
        Q + qidx * SQ[0] + h[:, None] * SQ[1] + d[None, :] * SQ[2],
        mask=(h[:, None] < (kh + 1) * gh) & (d[None, :] < D),
        other=0,
    )
    qs = tl.load(QScale) if Q_SCALE_PTR else QScale
    ks = tl.load(KScale) if K_SCALE_PTR else KScale
    vs = tl.load(VScale) if V_SCALE_PTR else VScale
    scale = SCALE * qs * 1.4426950408889634
    m = tl.full((BLOCK_H,), float("-inf"), tl.float32)
    norm = tl.zeros((BLOCK_H,), tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)
    # Match the indexer's valid-prefix contract; trailing top-k slots need
    # not be initialized when the number of logical blocks is below TOPK.
    real_topk = tl.maximum(0, tl.minimum(TOPK, (qabs + 128) // 128))
    for ti in range(split * CHUNK, tl.minimum((split + 1) * CHUNK, real_topk)):
        block = tl.load(TopK + kh * ST[0] + qidx * ST[1] + ti * ST[2])
        pos = block.to(tl.int64) * 128 + n
        page_col = pos // PAGE_SIZE
        valid = (block >= 0) & (block < (seq_len + 127) // 128)
        valid = valid & (pos >= 0) & (pos < seq_len) & (pos <= qabs)
        valid = valid & (page_col < PAGE_COLS)
        page = tl.load(Pages + req * SP[0] + page_col * SP[1], mask=valid, other=-1)
        valid = valid & (page >= 0) & (page < NUM_PAGES)
        page = tl.where(valid, page, 0).to(tl.int64)
        offset = pos % PAGE_SIZE
        k = tl.load(
            K
            + page[None, :] * SK[0]
            + offset[None, :] * SK[1]
            + kh * SK[2]
            + d[:, None] * SK[3],
            mask=valid[None, :] & (d[:, None] < D),
            other=0.0,
        )
        v = tl.load(
            V
            + page[:, None] * SV[0]
            + offset[:, None] * SV[1]
            + kh * SV[2]
            + d[None, :] * SV[3],
            mask=valid[:, None] & (d[None, :] < D),
            other=0.0,
        )
        k = (k.to(tl.float32) * ks).to(q.dtype)
        v = (v.to(tl.float32) * vs).to(q.dtype)
        scores = tl.dot(q, k) * scale
        scores = tl.where(valid[None, :], scores, float("-inf"))
        m_new = tl.maximum(m, tl.max(scores, 1))
        safe_m = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m - safe_m)
        p = tl.exp2(scores - safe_m[:, None])
        acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v)
        norm = norm * alpha + tl.sum(p, 1)
        m = m_new
    acc = acc / tl.where(norm > 0, norm, 1.0)[:, None]
    tl.store(
        Out + split * SO[0] + qidx * SO[1] + h[:, None] * SO[2] + d[None, :] * SO[3],
        acc,
        mask=(h[:, None] < (kh + 1) * gh) & (d[None, :] < D),
    )
    if SPLITS > 1:
        lse = tl.where(norm > 0, m + tl.log2(norm), float("-inf"))
        tl.store(
            Lse + split * SL[0] + qidx * SL[1] + h * SL[2], lse, mask=h < (kh + 1) * gh
        )


@triton.jit
def _merge_paged_sparse_kernel(
    Partial,
    Lse,
    Out,
    SPLITS: tl.constexpr,
    D: tl.constexpr,
    SP: tl.constexpr,
    SL: tl.constexpr,
    SO: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    s = tl.arange(0, SPLITS)
    d = tl.arange(0, BLOCK_D)
    lse = tl.load(Lse + s * SL[0] + row * SL[1] + head * SL[2])
    m = tl.max(lse, 0)
    safe_m = tl.where(m == float("-inf"), 0.0, m)
    w = tl.exp2(lse - safe_m)
    denom = tl.sum(w, 0)
    partial = tl.load(
        Partial + s[:, None] * SP[0] + row * SP[1] + head * SP[2] + d[None, :] * SP[3],
        mask=d[None, :] < D,
        other=0,
    )
    out = tl.sum(partial * w[:, None], 0) / tl.where(denom > 0, denom, 1.0)
    tl.store(Out + row * SO[0] + head * SO[1] + d * SO[2], out, mask=d < D)


def _tensor(tensor, name, shape, dtypes, device):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    if tensor.dtype not in dtypes:
        raise NotImplementedError(f"Unsupported {name} dtype: {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}")
    if tensor.requires_grad:
        raise NotImplementedError("Sparse attention is inference-only")
    if any(
        size > 1 and stride <= 0 for size, stride in zip(tensor.shape, tensor.stride())
    ):
        raise NotImplementedError(f"{name} requires positive non-broadcast strides")
    # A conservative sufficient condition for non-overlapping strides.
    span = 1
    for stride, size in sorted(
        (st, sz) for st, sz in zip(tensor.stride(), tensor.shape) if sz > 1
    ):
        if stride < span:
            raise NotImplementedError(f"{name} has overlapping strides")
        span += (size - 1) * stride


def _scale(value, name, device):
    if value is None:
        return 1.0, False
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or value.dtype != torch.float32 or value.device != device:
            raise ValueError(f"{name} must be a device FP32 scalar tensor")
        if value.requires_grad:
            raise NotImplementedError("Scale tensors must be inference-only")
        return value, True
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a scalar or device FP32 scalar tensor")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value), False


def _disjoint(output, inputs):
    if any(torch._C._overlaps(output, tensor) for tensor in inputs):
        raise ValueError("Output/workspace must not alias inputs or other workspace")


def _validate(q, k, v, topk, pages, seq, output, page_size, sm_scale):
    if not isinstance(q, torch.Tensor) or q.ndim != 3:
        raise ValueError("q must have shape [total_q, num_heads, head_dim]")
    if q.device.type != "cuda":
        raise NotImplementedError("Paged sparse attention requires a CUDA/ROCm device")
    t, h, d = q.shape
    if h <= 0 or d not in (64, 128, 256):
        raise NotImplementedError(
            "Positive head count and head_dim 64/128/256 are required"
        )
    if isinstance(page_size, bool) or not isinstance(page_size, int) or page_size <= 0:
        raise ValueError("page_size must be a positive integer")
    if not isinstance(k, torch.Tensor) or k.ndim != 4:
        raise ValueError(
            "k_cache must have shape [pages, page_size, kv_heads, head_dim]"
        )
    p, ps, hk, kd = k.shape
    if ps != page_size or kd != d or hk <= 0 or h % hk:
        raise ValueError("Invalid KV page size, head dimension or GQA head counts")
    if not isinstance(topk, torch.Tensor) or topk.ndim != 3:
        raise ValueError("topk_idx must have shape [kv_heads, total_q, topk]")
    if not isinstance(seq, torch.Tensor) or seq.ndim != 1:
        raise ValueError("seq_lens must be a vector")
    b = seq.shape[0]
    if not isinstance(pages, torch.Tensor) or pages.ndim != 2:
        raise ValueError("block_table must be a matrix")
    if t and (b == 0 or p == 0):
        raise ValueError("Non-empty queries require requests and allocated KV pages")
    if isinstance(sm_scale, bool) or not isinstance(sm_scale, (int, float)):
        raise TypeError("sm_scale must be a Python scalar")
    if not math.isfinite(sm_scale) or sm_scale <= 0:
        raise ValueError("sm_scale must be finite and positive")
    _tensor(q, "q", (t, h, d), (torch.float16, torch.bfloat16), q.device)
    _tensor(k, "k_cache", (p, ps, hk, d), (q.dtype,) + _FLOAT8, q.device)
    _tensor(v, "v_cache", k.shape, (k.dtype,), q.device)
    _tensor(topk, "topk_idx", (hk, t, topk.shape[-1]), (torch.int32,), q.device)
    _tensor(pages, "block_table", (b, pages.shape[-1]), (torch.int32,), q.device)
    _tensor(seq, "seq_lens", (b,), (torch.int32,), q.device)
    if b > 1 and seq.stride(0) != 1:
        raise NotImplementedError("seq_lens must be contiguous")
    if output is None:
        output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    _tensor(output, "output", q.shape, (q.dtype,), q.device)
    _disjoint(output, (q, k, v, topk, pages, seq))
    return output


def _splits(total_q, kv_heads, topk):
    target = max(1, min(topk, 256 // max(1, total_q * kv_heads)))
    return 1 << (target.bit_length() - 1)


def minimax_m3_sparse_decode_workspace(q, num_kv_heads, topk):
    """Allocate reusable FP32 partial/LSE buffers before Graph capture.

    Size depends only on tensor shapes. Caller must not share this workspace
    between concurrent streams or launches that may overlap.
    """
    if not isinstance(q, torch.Tensor) or q.ndim != 3:
        raise ValueError("q must have shape [total_q, num_heads, head_dim]")
    if (
        isinstance(num_kv_heads, bool)
        or not isinstance(num_kv_heads, int)
        or isinstance(topk, bool)
        or not isinstance(topk, int)
        or num_kv_heads <= 0
        or q.shape[1] % num_kv_heads
        or topk < 0
    ):
        raise ValueError("Invalid GQA head count or topk")
    _tensor(q, "q", q.shape, (torch.float16, torch.bfloat16), q.device)
    splits = _splits(q.shape[0], num_kv_heads, topk)
    return (
        torch.empty((splits,) + tuple(q.shape), dtype=torch.float32, device=q.device),
        torch.empty(
            (splits,) + tuple(q.shape[:2]), dtype=torch.float32, device=q.device
        ),
    )


def _launch(
    q,
    k,
    v,
    topk,
    pages,
    seq,
    cu,
    prefix,
    output,
    max_query_len,
    page_size,
    sm_scale,
    q_scale,
    k_scale,
    v_scale,
    decode_query_len,
    workspace,
):
    output = _validate(q, k, v, topk, pages, seq, output, page_size, sm_scale)
    qs, qp = _scale(q_scale, "q_scale", q.device)
    ks, kp = _scale(k_scale, "k_scale", q.device)
    vs, vp = _scale(v_scale, "v_scale", q.device)
    _disjoint(output, tuple(x for x in (qs, ks, vs) if isinstance(x, torch.Tensor)))
    if q.shape[0] == 0:
        return output
    decode = decode_query_len is not None
    t, h, d = q.shape
    hk = k.shape[2]
    splits = _splits(t, hk, topk.shape[-1]) if decode else 1
    if splits > 1:
        if workspace is None:
            workspace = minimax_m3_sparse_decode_workspace(q, hk, topk.shape[-1])
        if not isinstance(workspace, (tuple, list)) or len(workspace) != 2:
            raise ValueError("workspace must be (partial_output, partial_lse)")
        partial, lse = workspace
        _tensor(
            partial, "partial_output", (splits, t, h, d), (torch.float32,), q.device
        )
        _tensor(lse, "partial_lse", (splits, t, h), (torch.float32,), q.device)
        tensors = (q, k, v, topk, pages, seq, cu, prefix, output)
        tensors += tuple(x for x in (qs, ks, vs) if isinstance(x, torch.Tensor))
        _disjoint(partial, tensors + (lse,))
        _disjoint(lse, tensors)
    else:
        partial, lse = output.unsqueeze(0), output
    grid = (t, hk, splits) if decode else (max_query_len, seq.shape[0] * hk, 1)
    _paged_sparse_kernel[grid](
        q,
        k,
        v,
        topk,
        pages,
        cu,
        seq,
        prefix,
        partial,
        lse,
        qs,
        ks,
        vs,
        T=t,
        HQ=h,
        HK=hk,
        D=d,
        PAGE_SIZE=page_size,
        NUM_PAGES=k.shape[0],
        PAGE_COLS=pages.shape[1],
        TOPK=topk.shape[-1],
        DECODE=decode,
        QLEN=decode_query_len or 1,
        SPLITS=splits,
        CHUNK=triton.cdiv(topk.shape[-1], splits),
        SCALE=float(sm_scale),
        SQ=q.stride(),
        SK=k.stride(),
        SV=v.stride(),
        ST=topk.stride(),
        SP=pages.stride(),
        SO=partial.stride(),
        SL=lse.stride() if splits > 1 else (0, 0, 0),
        Q_SCALE_PTR=qp,
        K_SCALE_PTR=kp,
        V_SCALE_PTR=vp,
        BLOCK_H=max(16, triton.next_power_of_2(h // hk)),
        BLOCK_D=triton.next_power_of_2(d),
        num_warps=4,
        num_stages=1,
    )
    if splits > 1:
        _merge_paged_sparse_kernel[(t, h)](
            partial,
            lse,
            output,
            SPLITS=splits,
            D=d,
            SP=partial.stride(),
            SL=lse.stride(),
            SO=output.stride(),
            BLOCK_D=triton.next_power_of_2(d),
            num_warps=4,
        )
    return output


def minimax_m3_sparse_attn_paged(
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
    """Selected-block causal GQA prefill; return the supplied/new output.

    topk_idx contains logical 128-token block IDs. Device metadata must have
    monotone cu_seqlens_q starting at 0 and ending at total_q, query lengths
    <= max_query_len, and 0 <= prefix_len <= prefix_len + qlen <= seq_len.
    Metadata values are a caller contract; they are never read on the host.
    """
    if (
        isinstance(max_query_len, bool)
        or not isinstance(max_query_len, int)
        or max_query_len < 0
    ):
        raise ValueError("max_query_len must be a nonnegative integer")
    b = seq_lens.shape[0]
    _tensor(cu_seqlens_q, "cu_seqlens_q", (b + 1,), (torch.int32,), q.device)
    _tensor(prefix_lens, "prefix_lens", (b,), (torch.int32,), q.device)
    if cu_seqlens_q.stride(0) != 1 or (b > 1 and prefix_lens.stride(0) != 1):
        raise NotImplementedError("Query sequence metadata must be contiguous")
    if q.shape[0] and max_query_len == 0:
        raise ValueError("Non-empty queries require max_query_len > 0")
    if output is not None:
        _disjoint(output, (cu_seqlens_q, prefix_lens))
    return _launch(
        q,
        k_cache,
        v_cache,
        topk_idx,
        block_table,
        seq_lens,
        cu_seqlens_q,
        prefix_lens,
        output,
        max_query_len,
        page_size,
        sm_scale,
        q_scale,
        k_scale,
        v_scale,
        None,
        None,
    )


def minimax_m3_sparse_attn_decode_paged(
    q,
    k_cache,
    v_cache,
    topk_idx,
    block_table,
    seq_lens,
    sm_scale,
    output=None,
    decode_query_len=1,
    *,
    page_size=128,
    q_scale=None,
    k_scale=None,
    v_scale=None,
    workspace=None,
):
    """Uniform multi-token causal decode, with optional reusable workspace.

    seq_lens includes the decode queries. For each non-padding request it
    must be >= decode_query_len. Graph padding requests may have length 0.
    """
    if (
        isinstance(decode_query_len, bool)
        or not isinstance(decode_query_len, int)
        or decode_query_len <= 0
    ):
        raise ValueError("decode_query_len must be a positive integer")
    if q.shape[0] != seq_lens.shape[0] * decode_query_len:
        raise ValueError("total_q must equal num_requests * decode_query_len")
    return _launch(
        q,
        k_cache,
        v_cache,
        topk_idx,
        block_table,
        seq_lens,
        seq_lens,
        seq_lens,
        output,
        0,
        page_size,
        sm_scale,
        q_scale,
        k_scale,
        v_scale,
        decode_query_len,
        workspace,
    )
