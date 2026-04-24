"""
Triton Kernel for DSA Sparse Attention.

Flash-decoding style: split the TopK reduction across NUM_SPLITS programs to
parallelize a single token across many SMs. Phase-1 computes partial
(m, l, acc) per split; phase-2 combines them into the final output + LSE.
"""

import torch
import triton
import triton.language as tl


LOG2E = 1.4426950408889634


@triton.jit
def _split_attn_kernel(
    Q_nope_ptr, Q_pe_ptr,
    Ckv_ptr, Kpe_ptr,
    Indices_ptr,
    Partial_m_ptr, Partial_l_ptr, Partial_acc_ptr,
    sm_scale_log2e,
    stride_qn_t, stride_qn_h,
    stride_qp_t, stride_qp_h,
    stride_kc_s,
    stride_kp_s,
    stride_idx_t,
    stride_pm_t, stride_pm_s, stride_pm_h,
    stride_pl_t, stride_pl_s, stride_pl_h,
    stride_pacc_t, stride_pacc_s, stride_pacc_h, stride_pacc_d,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    D_CKV: tl.constexpr,
    D_KPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    t = tl.program_id(0)
    s = tl.program_id(1)

    SPLIT_SIZE: tl.constexpr = TOPK // NUM_SPLITS

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

    NEG_INF: tl.constexpr = float("-inf")
    m_i = tl.full([H], NEG_INF, dtype=tl.float32)
    l_i = tl.zeros([H], dtype=tl.float32)
    acc = tl.zeros([H, D_CKV], dtype=tl.float32)

    start = s * SPLIT_SIZE
    for bn in range(0, SPLIT_SIZE, BLOCK_N):
        idx_ptrs = Indices_ptr + t * stride_idx_t + (start + bn + offs_n)
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
        logits = tl.where(valid[None, :], logits, NEG_INF)

        m_new = tl.maximum(m_i, tl.max(logits, axis=1))
        # Guard against all-padding splits: -inf - -inf would be NaN.
        m_new_safe = tl.where(m_new == NEG_INF, 0.0, m_new)
        alpha = tl.where(m_i == NEG_INF, 0.0, tl.exp2(m_i - m_new_safe))
        p = tl.exp2(logits - m_new_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), kc, acc=acc)
        m_i = m_new

    pm_ptrs = Partial_m_ptr + t * stride_pm_t + s * stride_pm_s + offs_h * stride_pm_h
    pl_ptrs = Partial_l_ptr + t * stride_pl_t + s * stride_pl_s + offs_h * stride_pl_h
    pacc_ptrs = (
        Partial_acc_ptr
        + t * stride_pacc_t
        + s * stride_pacc_s
        + offs_h[:, None] * stride_pacc_h
        + offs_ckv[None, :] * stride_pacc_d
    )
    tl.store(pm_ptrs, m_i)
    tl.store(pl_ptrs, l_i)
    tl.store(pacc_ptrs, acc)


@triton.jit
def _combine_kernel(
    Partial_m_ptr, Partial_l_ptr, Partial_acc_ptr,
    Out_ptr, Lse_ptr,
    stride_pm_t, stride_pm_s, stride_pm_h,
    stride_pl_t, stride_pl_s, stride_pl_h,
    stride_pacc_t, stride_pacc_s, stride_pacc_h, stride_pacc_d,
    stride_out_t, stride_out_h,
    stride_lse_t,
    H: tl.constexpr,
    D_CKV: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    t = tl.program_id(0)

    offs_h = tl.arange(0, H)
    offs_ckv = tl.arange(0, D_CKV)

    NEG_INF: tl.constexpr = float("-inf")
    m_global = tl.full([H], NEG_INF, dtype=tl.float32)
    l_global = tl.zeros([H], dtype=tl.float32)
    acc = tl.zeros([H, D_CKV], dtype=tl.float32)

    for si in tl.static_range(NUM_SPLITS):
        pm_ptr = Partial_m_ptr + t * stride_pm_t + si * stride_pm_s + offs_h * stride_pm_h
        pl_ptr = Partial_l_ptr + t * stride_pl_t + si * stride_pl_s + offs_h * stride_pl_h
        m_si = tl.load(pm_ptr)
        l_si = tl.load(pl_ptr)

        pacc_ptr = (
            Partial_acc_ptr
            + t * stride_pacc_t
            + si * stride_pacc_s
            + offs_h[:, None] * stride_pacc_h
            + offs_ckv[None, :] * stride_pacc_d
        )
        acc_si = tl.load(pacc_ptr)

        m_new = tl.maximum(m_global, m_si)
        m_new_safe = tl.where(m_new == NEG_INF, 0.0, m_new)
        alpha = tl.where(m_global == NEG_INF, 0.0, tl.exp2(m_global - m_new_safe))
        beta = tl.where(m_si == NEG_INF, 0.0, tl.exp2(m_si - m_new_safe))
        acc = acc * alpha[:, None] + acc_si * beta[:, None]
        l_global = l_global * alpha + l_si * beta
        m_global = m_new

    l_safe = tl.where(l_global == 0.0, 1.0, l_global)
    acc = acc / l_safe[:, None]
    lse_val = m_global + tl.log2(l_global)

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
    NUM_SPLITS = 8

    device = q_nope.device
    partial_m = torch.empty((num_tokens, NUM_SPLITS, H), dtype=torch.float32, device=device)
    partial_l = torch.empty((num_tokens, NUM_SPLITS, H), dtype=torch.float32, device=device)
    partial_acc = torch.empty((num_tokens, NUM_SPLITS, H, D_ckv), dtype=torch.float32, device=device)

    grid1 = (num_tokens, NUM_SPLITS)
    _split_attn_kernel[grid1](
        q_nope, q_pe,
        ckv_flat, kpe_flat,
        sparse_indices,
        partial_m, partial_l, partial_acc,
        sm_scale * LOG2E,
        q_nope.stride(0), q_nope.stride(1),
        q_pe.stride(0), q_pe.stride(1),
        ckv_flat.stride(0),
        kpe_flat.stride(0),
        sparse_indices.stride(0),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
        partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
        TOPK=TOPK, H=H, D_CKV=D_ckv, D_KPE=D_kpe,
        BLOCK_N=BLOCK_N, NUM_SPLITS=NUM_SPLITS,
        num_warps=8, num_stages=2,
    )

    grid2 = (num_tokens,)
    _combine_kernel[grid2](
        partial_m, partial_l, partial_acc,
        output, lse,
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
        partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
        output.stride(0), output.stride(1),
        lse.stride(0),
        H=H, D_CKV=D_ckv, NUM_SPLITS=NUM_SPLITS,
        num_warps=4, num_stages=1,
    )
