# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Causal M3 GQA over separate paged K/V buffers.

Logical sparse blocks contain 128 tokens. Physical KV pages may have a
different size. Callers provide logical block IDs, physical page tables and
device sequence metadata; no cache packing or framework imports are needed.
The existing sparse_attn prefill/decode/merge kernels perform the attention;
this module only validates and dispatches the separate-cache interface.
"""

import math

import torch
import triton

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


def _kernel_module():
    # Lazy import keeps metadata-only calls independent of backend initialization.
    from . import sparse_attn

    return sparse_attn


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
    """Allocate reusable Q-dtype partials and FP32 LSE before Graph capture.

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
        torch.empty((splits,) + tuple(q.shape), dtype=q.dtype, device=q.device),
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
        _tensor(partial, "partial_output", (splits, t, h, d), (q.dtype,), q.device)
        _tensor(lse, "partial_lse", (splits, t, h), (torch.float32,), q.device)
        tensors = (q, k, v, topk, pages, seq, cu, prefix, output)
        tensors += tuple(x for x in (qs, ks, vs) if isinstance(x, torch.Tensor))
        _disjoint(partial, tensors + (lse,))
        _disjoint(lse, tensors)
    else:
        partial, lse = output.unsqueeze(0), output
    kernels = _kernel_module()
    layout = (
        page_size,
        k.shape[0],
        pages.shape[1],
        pages.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        v.stride(3),
    )
    options = dict(
        v_cache_ptr=v,
        KV_LAYOUT=layout,
        q_scale=qs,
        Q_SCALE_PTR=qp,
        K_SCALE_PTR=kp,
        V_SCALE_PTR=vp,
        BLOCK_SIZE_K=128,
        BLOCK_SIZE_D=triton.next_power_of_2(d),
        BLOCK_SIZE_H=max(16, triton.next_power_of_2(h // hk)),
        USE_FP8=k.dtype in _FLOAT8,
        KV_SCALE_MODE=1,
        num_warps=4,
        num_stages=1,
    )
    strides = (
        *q.stride(),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        k.stride(3),
        0,
        0,
        0,
        0,
        *topk.stride(),
    )
    if not decode:
        # Separate buffers cannot use the packed-KV TMA descriptor. Reuse the
        # existing pointer-based prefill path instead of packing the cache.
        kernels._gqa_sparse_fwd_fallback_kernel[(max_query_len, hk, seq.shape[0])](
            q,
            k,
            ks,
            vs,
            topk,
            output,
            pages,
            cu,
            cu,
            seq,
            prefix,
            hk,
            h // hk,
            d,
            topk.shape[-1],
            1,
            float(sm_scale),
            *strides,
            *output.stride(),
            pages.stride(0),
            BLOCK_SIZE_Q=1,
            BLOCK_SIZE_QH=options["BLOCK_SIZE_H"],
            **options,
        )
    else:
        # Keep the existing split-K and PDL scheduling. A single split writes
        # directly to output and does not need an LSE store or merge launch.
        use_pdl = splits > 1 and kernels.current_platform.is_arch_support_pdl()
        if use_pdl:
            options["launch_pdl"] = True
        kernels._gqa_sparse_decode_kernel[(t * splits, hk)](
            q,
            k,
            ks,
            vs,
            topk,
            partial,
            lse,
            pages,
            seq,
            t,
            h // hk,
            d,
            topk.shape[-1],
            float(sm_scale),
            decode_query_len,
            *strides,
            *partial.stride(),
            *(lse.stride() if splits > 1 else (0, 0, 0)),
            pages.stride(0),
            NUM_TOPK_CHUNKS=splits,
            USE_PDL=use_pdl,
            STORE_LSE=splits > 1,
            **options,
        )
        if splits > 1:
            kernels._merge_topk_attn_out_kernel[(t, h)](
                partial,
                lse,
                output,
                d,
                *partial.stride(),
                *lse.stride(),
                *output.stride(),
                NUM_TOPK_CHUNKS=splits,
                BLOCK_SIZE_D=options["BLOCK_SIZE_D"],
                USE_PDL=use_pdl,
                **({"launch_pdl": True} if use_pdl else {}),
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
