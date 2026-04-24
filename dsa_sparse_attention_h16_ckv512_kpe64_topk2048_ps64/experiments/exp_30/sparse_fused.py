"""
Triton Kernel for DSA Sparse Attention.

Hybrid dispatch:
  * T ≤ 2: single-launch D-parallel fused kernel (`_fused_attn_kernel`).
    Small per-CTA work; launch savings dominate.
  * T ≥ 3: single-launch "split+combine-in-one-kernel" kernel
    (`_fused_split_combine_kernel`). Uses an atomic-barrier between split
    and combine phases so the two 8-µs launch barriers collapse into one,
    while still preserving D-parallel combine across NUM_SPLITS CTAs per
    token (split index s becomes combine's d index after the barrier).
"""

import torch
import triton
import triton.language as tl


LOG2E = 1.4426950408889634


@triton.jit
def _fused_split_combine_kernel(
    Q_nope_ptr, Q_pe_ptr,
    Ckv_ptr, Kpe_ptr,
    Indices_ptr,
    Partial_m_ptr, Partial_l_ptr, Partial_acc_ptr,
    Counter_ptr,
    Out_ptr, Lse_ptr,
    sm_scale_log2e,
    stride_qn_t, stride_qn_h,
    stride_qp_t, stride_qp_h,
    stride_kc_s,
    stride_kp_s,
    stride_idx_t,
    stride_pm_t, stride_pm_s, stride_pm_h,
    stride_pl_t, stride_pl_s, stride_pl_h,
    stride_pacc_t, stride_pacc_s, stride_pacc_h, stride_pacc_d,
    stride_out_t, stride_out_h,
    stride_lse_t,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    D_CKV: tl.constexpr,
    D_KPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    t = tl.program_id(0)
    s = tl.program_id(1)

    SPLIT_SIZE: tl.constexpr = TOPK // NUM_SPLITS
    NEG_INF: tl.constexpr = float("-inf")

    offs_h = tl.arange(0, H)
    offs_ckv = tl.arange(0, D_CKV)
    offs_kpe = tl.arange(0, D_KPE)
    offs_n = tl.arange(0, BLOCK_N)

    # ===== Split phase =====
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

    m_i = tl.full([H], NEG_INF, dtype=tl.float32)
    l_i = tl.zeros([H], dtype=tl.float32)
    acc = tl.zeros([H, D_CKV], dtype=tl.float32)

    # Stride-partition: split `s` owns TopK positions {s, s+NUM_SPLITS, ...}
    # so a prefix-valid run of length N is spread across all NUM_SPLITS CTAs
    # as ~N/NUM_SPLITS each — straggler imbalance (one split doing all work)
    # is eliminated on small-valid workloads.
    offs_split = s + tl.arange(0, SPLIT_SIZE) * NUM_SPLITS
    idx_scan = tl.load(Indices_ptr + t * stride_idx_t + offs_split)
    num_valid = tl.sum((idx_scan >= 0).to(tl.int32), axis=0)
    max_bn = ((num_valid + BLOCK_N - 1) // BLOCK_N) * BLOCK_N

    for bn in range(0, max_bn, BLOCK_N):
        idx_ptrs = Indices_ptr + t * stride_idx_t + (s + (bn + offs_n) * NUM_SPLITS)
        idx = tl.load(idx_ptrs)
        valid = idx >= 0
        safe_idx = tl.where(valid, idx, 0).to(tl.int64)

        kc_ptrs = Ckv_ptr + safe_idx[:, None] * stride_kc_s + offs_ckv[None, :]
        kp_ptrs = Kpe_ptr + safe_idx[:, None] * stride_kp_s + offs_kpe[None, :]
        kc = tl.load(kc_ptrs, mask=valid[:, None], other=0.0, cache_modifier=".cg")
        kp = tl.load(kp_ptrs, mask=valid[:, None], other=0.0, cache_modifier=".cg")

        logits = tl.dot(q_nope, tl.trans(kc))
        logits = tl.dot(q_pe, tl.trans(kp), acc=logits)
        logits = logits * sm_scale_log2e
        logits = tl.where(valid[None, :], logits, NEG_INF)

        m_new = tl.maximum(m_i, tl.max(logits, axis=1))
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
    tl.store(pm_ptrs, m_i, cache_modifier=".cg")
    tl.store(pl_ptrs, l_i, cache_modifier=".cg")
    tl.store(pacc_ptrs, acc, cache_modifier=".cg")

    # ===== Atomic barrier =====
    # Release: all prior stores visible before the increment.
    tl.atomic_add(Counter_ptr + t, 1, sem="release")

    # Spin until all NUM_SPLITS CTAs have signalled done.
    # Using a plain `tl.load(..., volatile=True)` instead of
    # `atomic_add(0, sem="acquire")` avoids the L2 atomic-unit RMW
    # serialisation on the hot counter line — volatile loads are pure L2
    # reads. Producer's release on the increment still orders prior
    # partial_* stores on the B200's coherence point (L2). A
    # `tl.debug_barrier()` after the spin-exit prevents compiler
    # reordering of subsequent combine loads above the wait.
    count = tl.load(Counter_ptr + t, volatile=True)
    while count < NUM_SPLITS:
        count = tl.load(Counter_ptr + t, volatile=True)
    tl.debug_barrier()

    # ===== Combine phase (D-parallel; s indexes D-slice) =====
    d = s
    offs_d = d * BLOCK_D + tl.arange(0, BLOCK_D)

    m_global = tl.full([H], NEG_INF, dtype=tl.float32)
    l_global = tl.zeros([H], dtype=tl.float32)
    acc_comb = tl.zeros([H, BLOCK_D], dtype=tl.float32)

    for si in tl.static_range(NUM_SPLITS):
        pm_ptr_s = Partial_m_ptr + t * stride_pm_t + si * stride_pm_s + offs_h * stride_pm_h
        pl_ptr_s = Partial_l_ptr + t * stride_pl_t + si * stride_pl_s + offs_h * stride_pl_h
        m_si = tl.load(pm_ptr_s)
        l_si = tl.load(pl_ptr_s)

        pacc_ptr_s = (
            Partial_acc_ptr
            + t * stride_pacc_t
            + si * stride_pacc_s
            + offs_h[:, None] * stride_pacc_h
            + offs_d[None, :] * stride_pacc_d
        )
        acc_si = tl.load(pacc_ptr_s)

        m_new = tl.maximum(m_global, m_si)
        m_new_safe = tl.where(m_new == NEG_INF, 0.0, m_new)
        alpha = tl.where(m_global == NEG_INF, 0.0, tl.exp2(m_global - m_new_safe))
        beta = tl.where(m_si == NEG_INF, 0.0, tl.exp2(m_si - m_new_safe))
        acc_comb = acc_comb * alpha[:, None] + acc_si * beta[:, None]
        l_global = l_global * alpha + l_si * beta
        m_global = m_new

    l_safe = tl.where(l_global == 0.0, 1.0, l_global)
    acc_comb = acc_comb / l_safe[:, None]

    out_ptrs = (
        Out_ptr
        + t * stride_out_t
        + offs_h[:, None] * stride_out_h
        + offs_d[None, :]
    )
    tl.store(out_ptrs, acc_comb.to(tl.bfloat16))

    if d == 0:
        lse_val = m_global + tl.log2(l_global)
        lse_ptrs = Lse_ptr + t * stride_lse_t + offs_h
        tl.store(lse_ptrs, lse_val)

    # Decrement counter; once all NUM_SPLITS CTAs decrement, counter returns
    # to 0 for the next kernel call. Kernel-launch serialization guarantees
    # all decrements complete before the next call's split-phase atomic_add.
    tl.atomic_add(Counter_ptr + t, -1, sem="release")


@triton.jit
def _fused_attn_kernel(
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
    BLOCK_D: tl.constexpr,
):
    t = tl.program_id(0)
    d = tl.program_id(1)

    offs_h = tl.arange(0, H)
    offs_ckv = tl.arange(0, D_CKV)
    offs_kpe = tl.arange(0, D_KPE)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = d * BLOCK_D + tl.arange(0, BLOCK_D)

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
    acc = tl.zeros([H, BLOCK_D], dtype=tl.float32)

    offs_topk = tl.arange(0, TOPK)
    idx_scan = tl.load(Indices_ptr + t * stride_idx_t + offs_topk)
    num_valid = tl.sum((idx_scan >= 0).to(tl.int32), axis=0)
    max_bn = ((num_valid + BLOCK_N - 1) // BLOCK_N) * BLOCK_N

    for bn in range(0, max_bn, BLOCK_N):
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
        logits = tl.where(valid[None, :], logits, NEG_INF)

        m_new = tl.maximum(m_i, tl.max(logits, axis=1))
        m_new_safe = tl.where(m_new == NEG_INF, 0.0, m_new)
        alpha = tl.where(m_i == NEG_INF, 0.0, tl.exp2(m_i - m_new_safe))
        p = tl.exp2(logits - m_new_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        kc_slice_ptrs = Ckv_ptr + safe_idx[:, None] * stride_kc_s + offs_d[None, :]
        kc_slice = tl.load(kc_slice_ptrs, mask=valid[:, None], other=0.0)
        acc = tl.dot(p.to(tl.bfloat16), kc_slice, acc=acc)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]

    out_ptrs = (
        Out_ptr
        + t * stride_out_t
        + offs_h[:, None] * stride_out_h
        + offs_d[None, :]
    )
    tl.store(out_ptrs, acc.to(tl.bfloat16))

    if d == 0:
        lse_val = m_i + tl.log2(l_i)
        lse_ptrs = Lse_ptr + t * stride_lse_t + offs_h
        tl.store(lse_ptrs, lse_val)


# Cached counter tensors — one per (device, num_tokens) — kept at 0 between
# kernel calls by the in-kernel decrement at the end of the fused kernel.
_counter_cache: dict = {}


def _get_counter(num_tokens: int, device: torch.device) -> torch.Tensor:
    key = (device, num_tokens)
    cached = _counter_cache.get(key)
    if cached is None:
        cached = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        _counter_cache[key] = cached
    return cached


def kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    num_tokens, H, D_ckv = q_nope.shape
    D_kpe = q_pe.shape[-1]
    TOPK = sparse_indices.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape

    ckv_flat = ckv_cache.view(num_pages * page_size, D_ckv)
    kpe_flat = kpe_cache.view(num_pages * page_size, D_kpe)

    BLOCK_N = 128
    BLOCK_N_FUSED = 64
    D_CKV_SPLIT = 8
    BLOCK_D = D_ckv // D_CKV_SPLIT
    NUM_SPLITS = 8

    if num_tokens <= 2:
        grid = (num_tokens, D_CKV_SPLIT)
        _fused_attn_kernel[grid](
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
            BLOCK_N=BLOCK_N_FUSED, BLOCK_D=BLOCK_D,
            num_warps=8, num_stages=2,
        )
        return

    device = q_nope.device
    partial_m = torch.empty((num_tokens, NUM_SPLITS, H), dtype=torch.float32, device=device)
    partial_l = torch.empty((num_tokens, NUM_SPLITS, H), dtype=torch.float32, device=device)
    partial_acc = torch.empty((num_tokens, NUM_SPLITS, H, D_ckv), dtype=torch.float32, device=device)
    counter = _get_counter(num_tokens, device)

    assert NUM_SPLITS == D_CKV_SPLIT, (
        "fused split+combine kernel reuses program_id(1) as both split and "
        "D-slice index; NUM_SPLITS must equal D_CKV_SPLIT."
    )

    grid = (num_tokens, NUM_SPLITS)
    _fused_split_combine_kernel[grid](
        q_nope, q_pe,
        ckv_flat, kpe_flat,
        sparse_indices,
        partial_m, partial_l, partial_acc,
        counter,
        output, lse,
        sm_scale * LOG2E,
        q_nope.stride(0), q_nope.stride(1),
        q_pe.stride(0), q_pe.stride(1),
        ckv_flat.stride(0),
        kpe_flat.stride(0),
        sparse_indices.stride(0),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
        partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
        output.stride(0), output.stride(1),
        lse.stride(0),
        TOPK=TOPK, H=H, D_CKV=D_ckv, D_KPE=D_kpe,
        BLOCK_N=BLOCK_N, NUM_SPLITS=NUM_SPLITS, BLOCK_D=BLOCK_D,
        num_warps=8, num_stages=2,
    )
