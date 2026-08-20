"""
Triton Kernel for DSA Sparse Attention.

Fused flash-attention-style kernel with online softmax (base-2). One program
per query token: all 16 heads share the same TopK K/V, so they're processed
together (M=16 in tensor cores).
"""

import torch
import triton
import triton.language as tl


LOG2E = 1.4426950408889634


@triton.jit
def _fused_sparse_attn_kernel(
    Q_nope_ptr, Q_pe_ptr,
    Ckv_ptr, Kpe_ptr,
    Indices_ptr,
    Out_ptr, Lse_ptr,
    sm_scale_log2e,
    stride_qn_t, stride_qn_h,
    stride_qp_t, stride_qp_h,
    stride_kc_s,
    stride_kp_s,
    stride_idx_t,
    stride_out_t, stride_out_h,
    stride_lse_t,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    D_CKV: tl.constexpr,
    D_KPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    t = tl.program_id(0)

    offs_h = tl.arange(0, H)
    offs_ckv = tl.arange(0, D_CKV)
    offs_kpe = tl.arange(0, D_KPE)
    offs_n = tl.arange(0, BLOCK_N)

    q_nope_ptrs = (
        Q_nope_ptr
        + t * stride_qn_t
        + offs_h[:, None] * stride_qn_h
        + offs_ckv[None, :]
    )
    q_pe_ptrs = (
        Q_pe_ptr
        + t * stride_qp_t
        + offs_h[:, None] * stride_qp_h
        + offs_kpe[None, :]
    )
    q_nope = tl.load(q_nope_ptrs)
    q_pe = tl.load(q_pe_ptrs)

    m_i = tl.full([H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([H], dtype=tl.float32)
    acc = tl.zeros([H, D_CKV], dtype=tl.float32)

    for bn in range(0, TOPK, BLOCK_N):
        idx_ptrs = Indices_ptr + t * stride_idx_t + (bn + offs_n)
        idx = tl.load(idx_ptrs)
        valid = idx >= 0
        safe_idx = tl.where(valid, idx, 0).to(tl.int64)

        kc_ptrs = Ckv_ptr + safe_idx[:, None] * stride_kc_s + offs_ckv[None, :]
        kp_ptrs = Kpe_ptr + safe_idx[:, None] * stride_kp_s + offs_kpe[None, :]
        kc = tl.load(kc_ptrs, mask=valid[:, None], other=0.0)
        kp = tl.load(kp_ptrs, mask=valid[:, None], other=0.0)

        logits = tl.dot(q_nope, tl.trans(kc))
        logits = tl.dot(q_pe, tl.trans(kp), acc=logits)
        logits = logits * sm_scale_log2e
        logits = tl.where(valid[None, :], logits, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(logits, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(logits - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), kc, acc=acc)
        m_i = m_new

    acc = acc / l_i[:, None]
    lse_val = m_i + tl.log2(l_i)

    out_ptrs = (
        Out_ptr
        + t * stride_out_t
        + offs_h[:, None] * stride_out_h
        + offs_ckv[None, :]
    )
    tl.store(out_ptrs, acc.to(tl.bfloat16))
    lse_ptrs = Lse_ptr + t * stride_lse_t + offs_h
    tl.store(lse_ptrs, lse_val)


def kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    num_tokens, H, D_ckv = q_nope.shape
    D_kpe = q_pe.shape[-1]
    TOPK = sparse_indices.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape

    ckv_flat = ckv_cache.view(num_pages * page_size, D_ckv)
    kpe_flat = kpe_cache.view(num_pages * page_size, D_kpe)

    BLOCK_N = 64

    grid = (num_tokens,)
    _fused_sparse_attn_kernel[grid](
        q_nope, q_pe,
        ckv_flat, kpe_flat,
        sparse_indices,
        output, lse,
        sm_scale * LOG2E,
        q_nope.stride(0), q_nope.stride(1),
        q_pe.stride(0), q_pe.stride(1),
        ckv_flat.stride(0),
        kpe_flat.stride(0),
        sparse_indices.stride(0),
        output.stride(0), output.stride(1),
        lse.stride(0),
        TOPK=TOPK, H=H, D_CKV=D_ckv, D_KPE=D_kpe,
        BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=2,
    )
