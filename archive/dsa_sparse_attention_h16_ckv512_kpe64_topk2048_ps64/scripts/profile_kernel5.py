"""
Profile the NEW fused kernel structure (exp_15 atomic-barrier).

Since split+combine are fused, we cannot time phases by kernel boundary.
Instead, we build STUBBED variants of the fused kernel that let us back out
phase costs by subtraction:

  * T_full    — full fused kernel (split + barrier + combine)
  * T_splitonly — kernel with combine stubbed (early exit after barrier)
  * T_nobarrier — kernel with barrier stubbed (both atomic ops removed)
  * T_combineonly — kernel with split stubbed (pre-filled partial_*, skips dot)
  * T_prologue — kernel that only loads Q, writes zeros (pure launch+load baseline)

Derived phase times:
  split       = T_splitonly - T_prologue
  barrier     = T_full - T_nobarrier        (cost of atomic spin)
  combine     = T_combineonly - T_prologue
  dispatch    = T_prologue                  (launch overhead floor)
  useful_work = split + combine             (compute floor)

Also times the T≤2 fused path for context.

Launch:
    modal run scripts/profile_kernel5.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("flashinfer-profile5")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest", add_python="3.12")
    .apt_install("git", "wget", "build-essential", "cmake")
    .pip_install("huggingface_hub")
    .run_commands(
        "pip install --force-reinstall --upgrade "
        "git+https://github.com/flashinfer-ai/flashinfer-bench.git@main",
    )
    .pip_install("cupti-python")
)


@app.function(image=image, gpu="B200:1", timeout=1800, volumes={TRACE_SET_PATH: trace_volume})
def run_profile5() -> dict:
    import logging
    import math
    import torch
    import numpy as np
    import triton
    import triton.language as tl
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    LOG2E = 1.4426950408889634

    H = 16
    D_CKV = 512
    D_KPE = 64
    TOPK = 2048
    BLOCK_N = 128
    BLOCK_N_FUSED = 64
    NUM_SPLITS = 8
    D_CKV_SPLIT = 8
    BLOCK_D = D_CKV // D_CKV_SPLIT

    # ===================== Kernel variants =====================

    @triton.jit
    def _full_fused(
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

        q_nope_ptrs = (
            Q_nope_ptr + t * stride_qn_t + offs_h[:, None] * stride_qn_h + offs_ckv[None, :]
        )
        q_pe_ptrs = (
            Q_pe_ptr + t * stride_qp_t + offs_h[:, None] * stride_qp_h + offs_kpe[None, :]
        )
        q_nope = tl.load(q_nope_ptrs)
        q_pe = tl.load(q_pe_ptrs)

        m_i = tl.full([H], NEG_INF, dtype=tl.float32)
        l_i = tl.zeros([H], dtype=tl.float32)
        acc = tl.zeros([H, D_CKV], dtype=tl.float32)

        start = s * SPLIT_SIZE

        offs_split = tl.arange(0, SPLIT_SIZE)
        idx_scan = tl.load(Indices_ptr + t * stride_idx_t + start + offs_split)
        num_valid = tl.sum((idx_scan >= 0).to(tl.int32), axis=0)
        max_bn = ((num_valid + BLOCK_N - 1) // BLOCK_N) * BLOCK_N

        for bn in range(0, max_bn, BLOCK_N):
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
            Partial_acc_ptr + t * stride_pacc_t + s * stride_pacc_s
            + offs_h[:, None] * stride_pacc_h + offs_ckv[None, :] * stride_pacc_d
        )
        tl.store(pm_ptrs, m_i)
        tl.store(pl_ptrs, l_i)
        tl.store(pacc_ptrs, acc)

        tl.atomic_add(Counter_ptr + t, 1, sem="release")
        count = 0
        while count < NUM_SPLITS:
            count = tl.atomic_add(Counter_ptr + t, 0, sem="acquire")

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
                Partial_acc_ptr + t * stride_pacc_t + si * stride_pacc_s
                + offs_h[:, None] * stride_pacc_h + offs_d[None, :] * stride_pacc_d
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
            Out_ptr + t * stride_out_t + offs_h[:, None] * stride_out_h + offs_d[None, :]
        )
        tl.store(out_ptrs, acc_comb.to(tl.bfloat16))

        if d == 0:
            lse_val = m_global + tl.log2(l_global)
            lse_ptrs = Lse_ptr + t * stride_lse_t + offs_h
            tl.store(lse_ptrs, lse_val)

        tl.atomic_add(Counter_ptr + t, -1, sem="release")

    # -------- Variant: SPLIT ONLY (no barrier, no combine) --------
    @triton.jit
    def _split_only(
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
        NEG_INF: tl.constexpr = float("-inf")

        offs_h = tl.arange(0, H)
        offs_ckv = tl.arange(0, D_CKV)
        offs_kpe = tl.arange(0, D_KPE)
        offs_n = tl.arange(0, BLOCK_N)

        q_nope_ptrs = (
            Q_nope_ptr + t * stride_qn_t + offs_h[:, None] * stride_qn_h + offs_ckv[None, :]
        )
        q_pe_ptrs = (
            Q_pe_ptr + t * stride_qp_t + offs_h[:, None] * stride_qp_h + offs_kpe[None, :]
        )
        q_nope = tl.load(q_nope_ptrs)
        q_pe = tl.load(q_pe_ptrs)

        m_i = tl.full([H], NEG_INF, dtype=tl.float32)
        l_i = tl.zeros([H], dtype=tl.float32)
        acc = tl.zeros([H, D_CKV], dtype=tl.float32)

        start = s * SPLIT_SIZE

        offs_split = tl.arange(0, SPLIT_SIZE)
        idx_scan = tl.load(Indices_ptr + t * stride_idx_t + start + offs_split)
        num_valid = tl.sum((idx_scan >= 0).to(tl.int32), axis=0)
        max_bn = ((num_valid + BLOCK_N - 1) // BLOCK_N) * BLOCK_N

        for bn in range(0, max_bn, BLOCK_N):
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
            Partial_acc_ptr + t * stride_pacc_t + s * stride_pacc_s
            + offs_h[:, None] * stride_pacc_h + offs_ckv[None, :] * stride_pacc_d
        )
        tl.store(pm_ptrs, m_i)
        tl.store(pl_ptrs, l_i)
        tl.store(pacc_ptrs, acc)

    # -------- Variant: NO BARRIER (split + combine fused, but no atomic spin) --------
    # This assumes all CTAs ran split (UB after the first call, but timing-only
    # because we pre-fill partial_* before calling).
    @triton.jit
    def _no_barrier(
        Q_nope_ptr, Q_pe_ptr,
        Ckv_ptr, Kpe_ptr,
        Indices_ptr,
        Partial_m_ptr, Partial_l_ptr, Partial_acc_ptr,
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

        q_nope_ptrs = (
            Q_nope_ptr + t * stride_qn_t + offs_h[:, None] * stride_qn_h + offs_ckv[None, :]
        )
        q_pe_ptrs = (
            Q_pe_ptr + t * stride_qp_t + offs_h[:, None] * stride_qp_h + offs_kpe[None, :]
        )
        q_nope = tl.load(q_nope_ptrs)
        q_pe = tl.load(q_pe_ptrs)

        m_i = tl.full([H], NEG_INF, dtype=tl.float32)
        l_i = tl.zeros([H], dtype=tl.float32)
        acc = tl.zeros([H, D_CKV], dtype=tl.float32)

        start = s * SPLIT_SIZE

        offs_split = tl.arange(0, SPLIT_SIZE)
        idx_scan = tl.load(Indices_ptr + t * stride_idx_t + start + offs_split)
        num_valid = tl.sum((idx_scan >= 0).to(tl.int32), axis=0)
        max_bn = ((num_valid + BLOCK_N - 1) // BLOCK_N) * BLOCK_N

        for bn in range(0, max_bn, BLOCK_N):
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
            Partial_acc_ptr + t * stride_pacc_t + s * stride_pacc_s
            + offs_h[:, None] * stride_pacc_h + offs_ckv[None, :] * stride_pacc_d
        )
        tl.store(pm_ptrs, m_i)
        tl.store(pl_ptrs, l_i)
        tl.store(pacc_ptrs, acc)

        # No barrier — use pre-filled partial_* from prior call.
        # tl.debug_barrier()  # NOT a cross-CTA barrier; just stops reordering
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
                Partial_acc_ptr + t * stride_pacc_t + si * stride_pacc_s
                + offs_h[:, None] * stride_pacc_h + offs_d[None, :] * stride_pacc_d
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
            Out_ptr + t * stride_out_t + offs_h[:, None] * stride_out_h + offs_d[None, :]
        )
        tl.store(out_ptrs, acc_comb.to(tl.bfloat16))

        if d == 0:
            lse_val = m_global + tl.log2(l_global)
            lse_ptrs = Lse_ptr + t * stride_lse_t + offs_h
            tl.store(lse_ptrs, lse_val)

    # -------- Variant: COMBINE ONLY (skip split, use pre-filled partials) --------
    @triton.jit
    def _combine_only(
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
        BLOCK_D: tl.constexpr,
    ):
        t = tl.program_id(0)
        s = tl.program_id(1)
        NEG_INF: tl.constexpr = float("-inf")

        offs_h = tl.arange(0, H)

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
                Partial_acc_ptr + t * stride_pacc_t + si * stride_pacc_s
                + offs_h[:, None] * stride_pacc_h + offs_d[None, :] * stride_pacc_d
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
            Out_ptr + t * stride_out_t + offs_h[:, None] * stride_out_h + offs_d[None, :]
        )
        tl.store(out_ptrs, acc_comb.to(tl.bfloat16))

        if d == 0:
            lse_val = m_global + tl.log2(l_global)
            lse_ptrs = Lse_ptr + t * stride_lse_t + offs_h
            tl.store(lse_ptrs, lse_val)

    # -------- Variant: PROLOGUE ONLY (load Q only, no compute) --------
    @triton.jit
    def _prologue_only(
        Q_nope_ptr, Q_pe_ptr,
        Dummy_out_ptr,
        stride_qn_t, stride_qn_h,
        stride_qp_t, stride_qp_h,
        H: tl.constexpr,
        D_CKV: tl.constexpr,
        D_KPE: tl.constexpr,
    ):
        t = tl.program_id(0)
        s = tl.program_id(1)

        offs_h = tl.arange(0, H)
        offs_ckv = tl.arange(0, D_CKV)
        offs_kpe = tl.arange(0, D_KPE)

        q_nope_ptrs = (
            Q_nope_ptr + t * stride_qn_t + offs_h[:, None] * stride_qn_h + offs_ckv[None, :]
        )
        q_pe_ptrs = (
            Q_pe_ptr + t * stride_qp_t + offs_h[:, None] * stride_qp_h + offs_kpe[None, :]
        )
        q_nope = tl.load(q_nope_ptrs)
        q_pe = tl.load(q_pe_ptrs)
        # Write to dummy to force liveness
        if t == 9999:
            # never true; dead branch
            dst = Dummy_out_ptr + offs_h[:, None] * 0 + offs_ckv[None, :]
            tl.store(dst, q_nope.to(tl.bfloat16))
            dst = Dummy_out_ptr + offs_h[:, None] * 0 + offs_kpe[None, :]
            tl.store(dst, q_pe.to(tl.bfloat16))

    # -------- Variant: NOOP (pure launch) --------
    @triton.jit
    def _noop():
        pass

    # -------- Variant: FUSED (T<=2 path) --------
    @triton.jit
    def _fused_small(
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
            Q_nope_ptr + t * stride_qn_t + offs_h[:, None] * stride_qn_h + offs_ckv[None, :]
        )
        q_pe_ptrs = (
            Q_pe_ptr + t * stride_qp_t + offs_h[:, None] * stride_qp_h + offs_kpe[None, :]
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
            Out_ptr + t * stride_out_t + offs_h[:, None] * stride_out_h + offs_d[None, :]
        )
        tl.store(out_ptrs, acc.to(tl.bfloat16))

        if d == 0:
            lse_val = m_i + tl.log2(l_i)
            lse_ptrs = Lse_ptr + t * stride_lse_t + offs_h
            tl.store(lse_ptrs, lse_val)

    # ===================== Setup =====================

    from flashinfer_bench import TraceSet
    from safetensors import safe_open

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    DEF = "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64"
    workloads = trace_set.workloads.get(DEF, [])

    props = torch.cuda.get_device_properties(0)
    NUM_SM = props.multi_processor_count

    wls_by_T = {}
    for w in workloads:
        T = int(w.workload.axes.get("num_tokens"))
        wls_by_T.setdefault(T, []).append(w)

    def load_inputs(wrapped_wl):
        wl = wrapped_wl.workload
        T = int(wl.axes.get("num_tokens"))
        P = int(wl.axes.get("num_pages"))
        torch.manual_seed(0)
        device = torch.device("cuda")
        q_nope = torch.randn(T, 16, 512, dtype=torch.bfloat16, device=device)
        q_pe = torch.randn(T, 16, 64, dtype=torch.bfloat16, device=device)
        ckv = torch.randn(P, 64, 512, dtype=torch.bfloat16, device=device)
        kpe = torch.randn(P, 64, 64, dtype=torch.bfloat16, device=device)
        si_spec = wl.inputs["sparse_indices"]
        st_path = Path(trace_set.root) / si_spec.path
        with safe_open(str(st_path), framework="numpy") as f:
            si_np = f.get_tensor(si_spec.tensor_key)
        sparse_indices = torch.from_numpy(np.asarray(si_np)).to(device=device, dtype=torch.int32)
        sm_scale = 1.0 / math.sqrt(192)
        out = torch.empty(T, 16, 512, dtype=torch.bfloat16, device=device)
        lse = torch.empty(T, 16, dtype=torch.float32, device=device)
        return q_nope, q_pe, ckv, kpe, sparse_indices, sm_scale, out, lse, T, P

    def time_per_call_us(fn, warmup=50, inner=200, outer=50):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        xs = []
        for _ in range(outer):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(inner):
                fn()
            e.record()
            torch.cuda.synchronize()
            xs.append(s.elapsed_time(e) * 1000 / inner)
        xs.sort()
        return xs[int(0.5 * len(xs))], xs[int(0.9 * len(xs))]

    results = {
        "device": props.name,
        "num_sm": NUM_SM,
        "per_T": {},
    }

    # ---- Large T path measurements (T >= 3) ----
    for T in (6, 8):
        if T not in wls_by_T:
            continue
        q_nope, q_pe, ckv, kpe, si, sm_scale, out, lse, _, P = load_inputs(wls_by_T[T][0])
        valid_per_token = (si >= 0).sum(dim=-1).tolist()
        total_valid = int((si >= 0).sum().item())
        log.info(f"=== LARGE T={T}: valid={valid_per_token[:8]}..., total={total_valid} ===")

        device = q_nope.device
        ckv_flat = ckv.view(ckv.shape[0] * ckv.shape[1], ckv.shape[2])
        kpe_flat = kpe.view(kpe.shape[0] * kpe.shape[1], kpe.shape[2])

        partial_m = torch.empty((T, NUM_SPLITS, H), dtype=torch.float32, device=device)
        partial_l = torch.empty((T, NUM_SPLITS, H), dtype=torch.float32, device=device)
        partial_acc = torch.empty((T, NUM_SPLITS, H, D_CKV), dtype=torch.float32, device=device)
        counter = torch.zeros(T, dtype=torch.int32, device=device)
        dummy_out = torch.empty(1, dtype=torch.bfloat16, device=device)

        # Pre-fill partial_* with sensible values so combine sees real data
        partial_m.fill_(0.5)
        partial_l.fill_(1.0)
        partial_acc.uniform_(-1, 1)

        def run_full():
            _full_fused[(T, NUM_SPLITS)](
                q_nope, q_pe, ckv_flat, kpe_flat, si,
                partial_m, partial_l, partial_acc,
                counter, out, lse,
                sm_scale * LOG2E,
                q_nope.stride(0), q_nope.stride(1),
                q_pe.stride(0), q_pe.stride(1),
                ckv_flat.stride(0), kpe_flat.stride(0),
                si.stride(0),
                partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
                partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
                partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
                out.stride(0), out.stride(1),
                lse.stride(0),
                TOPK=TOPK, H=H, D_CKV=D_CKV, D_KPE=D_KPE,
                BLOCK_N=BLOCK_N, NUM_SPLITS=NUM_SPLITS, BLOCK_D=BLOCK_D,
                num_warps=8, num_stages=2,
            )

        def run_split_only():
            _split_only[(T, NUM_SPLITS)](
                q_nope, q_pe, ckv_flat, kpe_flat, si,
                partial_m, partial_l, partial_acc,
                sm_scale * LOG2E,
                q_nope.stride(0), q_nope.stride(1),
                q_pe.stride(0), q_pe.stride(1),
                ckv_flat.stride(0), kpe_flat.stride(0),
                si.stride(0),
                partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
                partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
                partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
                TOPK=TOPK, H=H, D_CKV=D_CKV, D_KPE=D_KPE,
                BLOCK_N=BLOCK_N, NUM_SPLITS=NUM_SPLITS,
                num_warps=8, num_stages=2,
            )

        def run_no_barrier():
            _no_barrier[(T, NUM_SPLITS)](
                q_nope, q_pe, ckv_flat, kpe_flat, si,
                partial_m, partial_l, partial_acc,
                out, lse,
                sm_scale * LOG2E,
                q_nope.stride(0), q_nope.stride(1),
                q_pe.stride(0), q_pe.stride(1),
                ckv_flat.stride(0), kpe_flat.stride(0),
                si.stride(0),
                partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
                partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
                partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
                out.stride(0), out.stride(1),
                lse.stride(0),
                TOPK=TOPK, H=H, D_CKV=D_CKV, D_KPE=D_KPE,
                BLOCK_N=BLOCK_N, NUM_SPLITS=NUM_SPLITS, BLOCK_D=BLOCK_D,
                num_warps=8, num_stages=2,
            )

        def run_combine_only():
            _combine_only[(T, D_CKV_SPLIT)](
                partial_m, partial_l, partial_acc, out, lse,
                partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
                partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
                partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
                out.stride(0), out.stride(1),
                lse.stride(0),
                H=H, D_CKV=D_CKV, NUM_SPLITS=NUM_SPLITS, BLOCK_D=BLOCK_D,
                num_warps=8, num_stages=2,
            )

        def run_prologue():
            _prologue_only[(T, NUM_SPLITS)](
                q_nope, q_pe, dummy_out,
                q_nope.stride(0), q_nope.stride(1),
                q_pe.stride(0), q_pe.stride(1),
                H=H, D_CKV=D_CKV, D_KPE=D_KPE,
                num_warps=8, num_stages=2,
            )

        def run_noop():
            _noop[(T, NUM_SPLITS)]()

        # Warmup + compile
        run_full()
        run_split_only()
        run_no_barrier()
        run_combine_only()
        run_prologue()
        run_noop()
        torch.cuda.synchronize()

        full_p50, full_p90 = time_per_call_us(run_full)
        split_p50, split_p90 = time_per_call_us(run_split_only)
        nobar_p50, nobar_p90 = time_per_call_us(run_no_barrier)
        comb_p50, comb_p90 = time_per_call_us(run_combine_only)
        prol_p50, prol_p90 = time_per_call_us(run_prologue)
        noop_p50, _ = time_per_call_us(run_noop)

        # Memcpy floor: size of partial_acc
        pacc_bytes = T * NUM_SPLITS * H * D_CKV * 4
        buf_r = torch.empty(pacc_bytes // 4, dtype=torch.float32, device=device)
        buf_w = torch.empty(pacc_bytes // 4, dtype=torch.float32, device=device)
        def memcpy_pacc():
            buf_w.copy_(buf_r)
        mem_p50, _ = time_per_call_us(memcpy_pacc)

        # Derived
        barrier_cost = full_p50 - nobar_p50  # cost of atomic spin + barrier
        split_work = split_p50 - prol_p50
        combine_work = comb_p50 - prol_p50

        log.info(f"T={T}: full={full_p50:.2f} nobar={nobar_p50:.2f} split_only={split_p50:.2f} combine_only={comb_p50:.2f} prologue={prol_p50:.2f} noop={noop_p50:.2f} memcpy_pacc={mem_p50:.2f}")
        log.info(f"T={T}: barrier={barrier_cost:.2f} split_work={split_work:.2f} combine_work={combine_work:.2f}")

        results["per_T"][T] = {
            "regime": "large",
            "valid_per_token": valid_per_token,
            "total_valid": total_valid,
            "full_p50_us": full_p50, "full_p90_us": full_p90,
            "split_only_p50_us": split_p50, "split_only_p90_us": split_p90,
            "no_barrier_p50_us": nobar_p50, "no_barrier_p90_us": nobar_p90,
            "combine_only_p50_us": comb_p50, "combine_only_p90_us": comb_p90,
            "prologue_p50_us": prol_p50, "prologue_p90_us": prol_p90,
            "noop_p50_us": noop_p50,
            "memcpy_pacc_p50_us": mem_p50,
            "barrier_cost_us": barrier_cost,
            "split_work_us": split_work,
            "combine_work_us": combine_work,
            "pacc_bytes": pacc_bytes,
            "grid": (T, NUM_SPLITS),
            "ctas": T * NUM_SPLITS,
            "occupancy_pct": 100.0 * T * NUM_SPLITS / NUM_SM,
        }

    # ---- Small T path measurements (T <= 2) ----
    for T in (1, 2):
        if T not in wls_by_T:
            continue
        q_nope, q_pe, ckv, kpe, si, sm_scale, out, lse, _, P = load_inputs(wls_by_T[T][0])
        valid_per_token = (si >= 0).sum(dim=-1).tolist()
        total_valid = int((si >= 0).sum().item())
        log.info(f"=== SMALL T={T}: valid={valid_per_token}, total={total_valid} ===")

        device = q_nope.device
        ckv_flat = ckv.view(ckv.shape[0] * ckv.shape[1], ckv.shape[2])
        kpe_flat = kpe.view(kpe.shape[0] * kpe.shape[1], kpe.shape[2])
        dummy_out = torch.empty(1, dtype=torch.bfloat16, device=device)

        def run_fused():
            _fused_small[(T, D_CKV_SPLIT)](
                q_nope, q_pe, ckv_flat, kpe_flat, si,
                out, lse,
                sm_scale * LOG2E,
                q_nope.stride(0), q_nope.stride(1),
                q_pe.stride(0), q_pe.stride(1),
                ckv_flat.stride(0), kpe_flat.stride(0),
                si.stride(0),
                out.stride(0), out.stride(1),
                lse.stride(0),
                TOPK=TOPK, H=H, D_CKV=D_CKV, D_KPE=D_KPE,
                BLOCK_N=BLOCK_N_FUSED, BLOCK_D=BLOCK_D,
                num_warps=8, num_stages=2,
            )

        def run_prologue_small():
            _prologue_only[(T, D_CKV_SPLIT)](
                q_nope, q_pe, dummy_out,
                q_nope.stride(0), q_nope.stride(1),
                q_pe.stride(0), q_pe.stride(1),
                H=H, D_CKV=D_CKV, D_KPE=D_KPE,
                num_warps=8, num_stages=2,
            )

        def run_noop_small():
            _noop[(T, D_CKV_SPLIT)]()

        run_fused()
        run_prologue_small()
        run_noop_small()
        torch.cuda.synchronize()

        fused_p50, fused_p90 = time_per_call_us(run_fused)
        prol_p50, prol_p90 = time_per_call_us(run_prologue_small)
        noop_p50, _ = time_per_call_us(run_noop_small)

        log.info(f"T={T} (small): fused={fused_p50:.2f} prologue={prol_p50:.2f} noop={noop_p50:.2f}")

        results["per_T"][T] = {
            "regime": "small",
            "valid_per_token": valid_per_token,
            "total_valid": total_valid,
            "fused_p50_us": fused_p50,
            "fused_p90_us": fused_p90,
            "prologue_p50_us": prol_p50,
            "noop_p50_us": noop_p50,
            "grid": (T, D_CKV_SPLIT),
            "ctas": T * D_CKV_SPLIT,
        }

    # ---- Full-path event-time profile across ALL workloads ----
    # For each workload, record the full-path event-time.
    all_rows = []
    for wrapped in workloads:
        q_nope, q_pe, ckv, kpe, si, sm_scale, out, lse, T, P = load_inputs(wrapped)
        valid = int((si >= 0).sum().item())
        uuid = wrapped.workload.uuid[:8]

        device = q_nope.device
        ckv_flat = ckv.view(ckv.shape[0] * ckv.shape[1], ckv.shape[2])
        kpe_flat = kpe.view(kpe.shape[0] * kpe.shape[1], kpe.shape[2])

        if T <= 2:
            def run():
                _fused_small[(T, D_CKV_SPLIT)](
                    q_nope, q_pe, ckv_flat, kpe_flat, si,
                    out, lse,
                    sm_scale * LOG2E,
                    q_nope.stride(0), q_nope.stride(1),
                    q_pe.stride(0), q_pe.stride(1),
                    ckv_flat.stride(0), kpe_flat.stride(0),
                    si.stride(0),
                    out.stride(0), out.stride(1),
                    lse.stride(0),
                    TOPK=TOPK, H=H, D_CKV=D_CKV, D_KPE=D_KPE,
                    BLOCK_N=BLOCK_N_FUSED, BLOCK_D=BLOCK_D,
                    num_warps=8, num_stages=2,
                )
        else:
            partial_m = torch.empty((T, NUM_SPLITS, H), dtype=torch.float32, device=device)
            partial_l = torch.empty((T, NUM_SPLITS, H), dtype=torch.float32, device=device)
            partial_acc = torch.empty((T, NUM_SPLITS, H, D_CKV), dtype=torch.float32, device=device)
            counter = torch.zeros(T, dtype=torch.int32, device=device)

            def run():
                _full_fused[(T, NUM_SPLITS)](
                    q_nope, q_pe, ckv_flat, kpe_flat, si,
                    partial_m, partial_l, partial_acc,
                    counter, out, lse,
                    sm_scale * LOG2E,
                    q_nope.stride(0), q_nope.stride(1),
                    q_pe.stride(0), q_pe.stride(1),
                    ckv_flat.stride(0), kpe_flat.stride(0),
                    si.stride(0),
                    partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
                    partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
                    partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
                    out.stride(0), out.stride(1),
                    lse.stride(0),
                    TOPK=TOPK, H=H, D_CKV=D_CKV, D_KPE=D_KPE,
                    BLOCK_N=BLOCK_N, NUM_SPLITS=NUM_SPLITS, BLOCK_D=BLOCK_D,
                    num_warps=8, num_stages=2,
                )

        try:
            run()
            torch.cuda.synchronize()
            p50, p90 = time_per_call_us(run, warmup=20, inner=100, outer=20)
            all_rows.append({"uuid": uuid, "T": T, "valid": valid, "p50_us": p50, "p90_us": p90})
        except Exception as exc:
            log.info(f"failed {uuid} T={T}: {exc}")
            continue

    results["all_workloads"] = all_rows
    return results


@app.local_entrypoint()
def main():
    print(f"Profiling fused kernel (exp_15) on Modal B200...")
    out = run_profile5.remote()
    import json
    print("\n=== FULL RESULT ===")
    print(json.dumps(out, indent=2, default=str))
