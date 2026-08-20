"""
Triton + Gluon Kernel for DSA Sparse Attention.

Implements batched Native Sparse Attention (DSA) with sparse TopK KV cache selection
for DeepSeek-V3 with tensor parallel size 8.

Split-K flash-decode design with tail-padding liveness skipping. The TOPK=2048
sparse indices are divided across NSPLIT CTAs per token (T <= 8 in this regime, so
splits restore SM occupancy). Padding (-1) is tail-only, so a chunk whose FIRST
index is -1 is entirely dead: such split CTAs exit before touching Q/KV (saves
DRAM contention), and the combine pass masks dead splits' partials out (median row
has ~33 valid of 2048 -> reads 1 split instead of 16). All loop bounds stay static
(dynamic trip counts cost +30% here — see exp_5). All 16 query heads share the
sparse KV set (MLA latent cache), forming the M dimension of the tensor-core dots.
Online softmax runs in the exp2 domain (base-2 LSE for free).

exp_24: the combine kernel is rewritten in Gluon. The Triton combine was
latency-bound: its merge loop pipelines only num_stages=2 tiles ahead, exposing
~8 DRAM round-trips over 16 split partials. The Gluon version issues cp.async
for ALL NSPLIT [H, DCB] pacc tiles up front (128 KB of smem in flight at once),
then overlaps the m/l reduction with the transfers and consumes tiles in commit
order.

exp_25: decode is rewritten in Gluon too. Both gather iterations (CHUNK=128 =
2 x BLOCK_N=64) issue their K-gathers as cp.async into double-buffered smem
BEFORE any consumption (explicit version of what Triton's num_stages=2
pipeliner did). Dots run on mma_v2 tensor cores (m16n8k16: H=16 fills M
exactly); kc tiles are read from smem twice — transposed for QK^T, straight
for PV — which is the smem-staging pattern Triton used internally.

exp_26: single launch. The combine pass is fused into the decode kernel as a
last-CTA merge tail: every split CTA (dead ones included) bumps a per-token
atomic counter (acq_rel) after its stores; the CTA that sees the final count
merges all alive splits' partials for the whole [H, DC] row and writes
out/lse. Counters live in grow-only workspace, are zeroed once at allocation,
and are never reset — each call adds exactly NSPLIT per token, so "last" is
`old % NSPLIT == NSPLIT - 1`. This deletes the second kernel launch (~6 us,
24-44% of total call latency per profile.md) on every workload.
"""

import math

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia import ampere
from triton.experimental.gluon.language.nvidia.ampere import async_copy

LOG2E = 1.4426950408889634

# Grow-only scratch workspace for split partials (exp_17 plan). Pure scratch:
# every byte read by the merge tail is written earlier in the same call, so
# reuse across calls is safe. Avoids per-call torch.empty() host overhead.
# cnt is zero-initialized ONCE at allocation and never reset (see module doc).
_WS = {"pacc": None, "pml": None, "cnt": None}


def _workspace(n_pacc, n_pml, n_cnt, device):
    pacc = _WS["pacc"]
    if pacc is None or pacc.numel() < n_pacc or pacc.device != device:
        pacc = torch.empty(n_pacc, dtype=torch.float32, device=device)
        _WS["pacc"] = pacc
    pml = _WS["pml"]
    if pml is None or pml.numel() < n_pml or pml.device != device:
        pml = torch.empty(n_pml, dtype=torch.float32, device=device)
        _WS["pml"] = pml
    cnt = _WS["cnt"]
    if cnt is None or cnt.numel() < n_cnt or cnt.device != device:
        cnt = torch.zeros(n_cnt, dtype=torch.int32, device=device)
        _WS["cnt"] = cnt
    return pacc, pml, cnt


@gluon.jit
def _dsa_decode_gluon(
    q_nope_ptr, q_pe_ptr, ckv_ptr, kpe_ptr, idx_ptr,
    out_ptr, lse_ptr,
    pacc_ptr, pml_ptr, cnt_ptr,
    qk_scale,  # sm_scale * log2(e)
    stride_qn_t, stride_qn_h,
    stride_qp_t, stride_qp_h,
    stride_idx_t,
    stride_o_t, stride_o_h,
    stride_l_t,
    NSPLIT: gl.constexpr,
    H: gl.constexpr, DC: gl.constexpr, DP: gl.constexpr,
    TOPK: gl.constexpr, BLOCK_N: gl.constexpr,
    MERGE_PAR: gl.constexpr,
):
    t = gl.program_id(0)
    s = gl.program_id(1)

    CHUNK: gl.constexpr = TOPK // NSPLIT
    NITER: gl.constexpr = CHUNK // BLOCK_N
    idx_base = idx_ptr + t * stride_idx_t + s * CHUNK

    # m16n8k16 MMA: H=16 fills M exactly. One layout serves logits [H, BLOCK_N]
    # and acc [H, DC] so the softmax scalars broadcast without converts.
    # kWidth=2 matches the native HMMA fragment packing: operand reads from
    # smem become ldmatrix, and the p -> A-operand convert is register-local
    # (kWidth=8 caused scalar ld.shared + smem round-trips + 68 reg spills).
    MMA: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0],
                                                  warps_per_cta=[1, 4],
                                                  instr_shape=[16, 8])
    DOTA: gl.constexpr = gl.DotOperandLayout(0, MMA, 2)
    DOTB: gl.constexpr = gl.DotOperandLayout(1, MMA, 2)
    KSL: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128,
                                             element_bitwidth=16, rank=2)
    # Gather pointer layouts: 8 contiguous bf16 per thread (16B cp.async).
    GLD: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0])
    GLDP: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    QLD: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0])
    KB: gl.constexpr = 128  # QK dot K-chunk: bounds live a-fragment registers

    qn_smem = gl.allocate_shared_memory(gl.bfloat16, [H, DC], KSL)
    qp_smem = gl.allocate_shared_memory(gl.bfloat16, [H, DP], KSL)
    kc_smem = gl.allocate_shared_memory(gl.bfloat16, [NITER, BLOCK_N, DC], KSL)
    kp_smem = gl.allocate_shared_memory(gl.bfloat16, [NITER, BLOCK_N, DP], KSL)

    # t-safe loads issue BEFORE the liveness check: Q rows and the chunk's
    # first idx block are always in bounds, and none of them depend on
    # `first`. This takes the liveness scalar's DRAM round-trip off the
    # critical path (it previously gated the Q-load and gather-issue chains).
    # Dead CTAs pull ~18KB of Q redundantly, but all 16 CTAs of a token read
    # the same lines — one DRAM fetch, 15 L2 hits.
    hq = gl.arange(0, H, layout=gl.SliceLayout(1, QLD))
    dcq = gl.arange(0, DC, layout=gl.SliceLayout(0, QLD))
    dpq = gl.arange(0, DP, layout=gl.SliceLayout(0, QLD))
    qn = gl.load(q_nope_ptr + t * stride_qn_t + hq[:, None] * stride_qn_h
                 + dcq[None, :])
    qp = gl.load(q_pe_ptr + t * stride_qp_t + hq[:, None] * stride_qp_h
                 + dpq[None, :])
    nc = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, GLD))
    dc = gl.arange(0, DC, layout=gl.SliceLayout(0, GLD))
    np_ = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, GLDP))
    dp = gl.arange(0, DP, layout=gl.SliceLayout(0, GLDP))
    idx_c0 = gl.load(idx_base + nc)
    idx_p0 = gl.load(idx_base + np_)
    # Row-solo predicate (idx[t, CHUNK] < 0 means <=1 live split). Read once
    # in the cycle-0 burst; serves both the split-0 direct-write epilogue and
    # the merge tail's skip check (always in bounds: CHUNK < TOPK).
    row_multi = gl.load(idx_ptr + t * stride_idx_t + CHUNK) >= 0

    # Tail-only padding: first index -1 => whole chunk dead. Split 0 always
    # runs (and stores) so the merge has a defined source even for empty rows.
    # Dead CTAs skip the body but still arrive at the counter below.
    first = gl.load(idx_base)
    if (s == 0) | (first >= 0):
        # Q goes through smem like every other MMA operand (holding the [H, DC]
        # A-fragment in registers costs 128 regs/thread and spills).
        qn_smem.store(qn)
        qp_smem.store(qp)

        # Issue BOTH iterations' K-gathers up front into double-buffered smem.
        # Tail-only padding: a block whose first index is -1 is entirely dead
        # (61% of rows have <64 valid entries — the whole second block is
        # padding). Skip its gather; commit an empty group so the static
        # wait_group bookkeeping below is unchanged.
        for it in gl.static_range(NITER):
            if it == 0:
                tok_c = gl.where(idx_c0 >= 0, idx_c0, 0)
                async_copy.async_copy_global_to_shared(
                    kc_smem.index(0), ckv_ptr + tok_c[:, None] * DC + dc[None, :])
                tok_p = gl.where(idx_p0 >= 0, idx_p0, 0)
                async_copy.async_copy_global_to_shared(
                    kp_smem.index(0), kpe_ptr + tok_p[:, None] * DP + dp[None, :])
            else:
                blk_alive = gl.load(idx_base + it * BLOCK_N) >= 0
                if blk_alive:
                    idx_c = gl.load(idx_base + it * BLOCK_N + nc)
                    tok_c = gl.where(idx_c >= 0, idx_c, 0)
                    async_copy.async_copy_global_to_shared(
                        kc_smem.index(it),
                        ckv_ptr + tok_c[:, None] * DC + dc[None, :])
                    idx_p = gl.load(idx_base + it * BLOCK_N + np_)
                    tok_p = gl.where(idx_p >= 0, idx_p, 0)
                    async_copy.async_copy_global_to_shared(
                        kp_smem.index(it),
                        kpe_ptr + tok_p[:, None] * DP + dp[None, :])
            async_copy.commit_group()

        h = gl.arange(0, H, layout=gl.SliceLayout(1, MMA))
        n_mma = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, MMA))
        dc_mma = gl.arange(0, DC, layout=gl.SliceLayout(0, MMA))

        m_i = gl.full([H], float("-inf"), gl.float32, gl.SliceLayout(1, MMA))
        l_i = gl.zeros([H], gl.float32, gl.SliceLayout(1, MMA))
        acc = gl.zeros([H, DC], gl.float32, MMA)

        for it in gl.static_range(NITER):
            if it == 0:
                async_copy.wait_group(NITER - 1)
                kc_s = kc_smem.index(0)
                kc_t = kc_s.permute([1, 0])                       # [DC, BLOCK_N]
                logits = gl.zeros([H, BLOCK_N], gl.float32, MMA)
                for kb in gl.static_range(DC // KB):
                    a = qn_smem.slice(kb * KB, KB, dim=1).load(DOTA)
                    b = kc_t.slice(kb * KB, KB, dim=0).load(DOTB)
                    logits = ampere.mma_v2(a, b, logits)
                qp_a = qp_smem.load(DOTA)                         # [H, DP]
                kp_t = kp_smem.index(0).permute([1, 0]).load(DOTB)
                logits = ampere.mma_v2(qp_a, kp_t, logits)        # [H, BLOCK_N]
                logits = logits * qk_scale
                valid = gl.load(idx_base + n_mma) >= 0
                logits = gl.where(valid[None, :], logits, float("-inf"))

                m_new = gl.maximum(m_i, gl.max(logits, 1))
                m_safe = gl.where(m_new == float("-inf"), 0.0, m_new)
                p = gl.exp2(logits - m_safe[:, None])             # [H, BLOCK_N]
                alpha = gl.exp2(m_i - m_safe)                     # 0 if m_i=-inf
                l_i = l_i * alpha + gl.sum(p, 1)
                pa = gl.convert_layout(p.to(gl.bfloat16), DOTA)
                kc_b = kc_s.load(DOTB)                            # [BLOCK_N, DC]
                acc = ampere.mma_v2(pa, kc_b, acc * alpha[:, None])
                m_i = m_new
            else:
                blk_alive = gl.load(idx_base + it * BLOCK_N) >= 0
                if blk_alive:
                    async_copy.wait_group(NITER - 1 - it)
                    kc_s = kc_smem.index(it)
                    kc_t = kc_s.permute([1, 0])                   # [DC, BLOCK_N]
                    logits = gl.zeros([H, BLOCK_N], gl.float32, MMA)
                    for kb in gl.static_range(DC // KB):
                        a = qn_smem.slice(kb * KB, KB, dim=1).load(DOTA)
                        b = kc_t.slice(kb * KB, KB, dim=0).load(DOTB)
                        logits = ampere.mma_v2(a, b, logits)
                    qp_a = qp_smem.load(DOTA)                     # [H, DP]
                    kp_t = kp_smem.index(it).permute([1, 0]).load(DOTB)
                    logits = ampere.mma_v2(qp_a, kp_t, logits)    # [H, BLOCK_N]
                    logits = logits * qk_scale
                    valid = gl.load(idx_base + it * BLOCK_N + n_mma) >= 0
                    logits = gl.where(valid[None, :], logits, float("-inf"))

                    m_new = gl.maximum(m_i, gl.max(logits, 1))
                    m_safe = gl.where(m_new == float("-inf"), 0.0, m_new)
                    p = gl.exp2(logits - m_safe[:, None])         # [H, BLOCK_N]
                    alpha = gl.exp2(m_i - m_safe)                 # 0 if m_i=-inf
                    l_i = l_i * alpha + gl.sum(p, 1)
                    pa = gl.convert_layout(p.to(gl.bfloat16), DOTA)
                    kc_b = kc_s.load(DOTB)                        # [BLOCK_N, DC]
                    acc = ampere.mma_v2(pa, kc_b, acc * alpha[:, None])
                    m_i = m_new

        if NSPLIT == 1:
            l_safe = gl.where(l_i == 0.0, 1.0, l_i)
            out = acc / l_safe[:, None]
            gl.store(out_ptr + t * stride_o_t + h[:, None] * stride_o_h
                     + dc_mma[None, :], out.to(out_ptr.dtype.element_ty))
            lse = gl.where(l_i == 0.0, float("-inf"), m_i + gl.log2(l_safe))
            gl.store(lse_ptr + t * stride_l_t + h, lse)
        else:
            # Solo split (s==0 and the next chunk is dead => this is the only
            # live split): write final results directly, skip partials; the
            # merge tail applies the same rule. ~70% of rows take this path.
            solo = (s == 0) & (row_multi == 0)
            if solo:
                l_safe = gl.where(l_i == 0.0, 1.0, l_i)
                out = acc / l_safe[:, None]
                gl.store(out_ptr + t * stride_o_t + h[:, None] * stride_o_h
                         + dc_mma[None, :], out.to(out_ptr.dtype.element_ty))
                lse = gl.where(l_i == 0.0, float("-inf"), m_i + gl.log2(l_safe))
                gl.store(lse_ptr + t * stride_l_t + h, lse)
            else:
                # Unnormalized partials: acc [H, DC] + (m, l) as two [H] rows.
                pacc_off = (t * NSPLIT + s) * H * DC
                gl.store(pacc_ptr + pacc_off + h[:, None] * DC + dc_mma[None, :],
                         acc)
                pml_off = (t * NSPLIT + s) * 2 * H
                gl.store(pml_ptr + pml_off + h, m_i)
                gl.store(pml_ptr + pml_off + H + h, l_i)

    if NSPLIT != 1:
        # Fused merge: the last MERGE_PAR split CTAs of this token (dead or
        # alive) each merge one DC slice. acq_rel pairs producers' stores with
        # the mergers' loads. Counters are monotonic (never reset): each call
        # adds exactly NSPLIT per token, so the call's arrival number is
        # old % NSPLIT. Late arrivals spin (acquire RMW) until all NSPLIT CTAs
        # arrived — safe only when the whole grid is co-resident, which the
        # host guarantees by setting MERGE_PAR=1 (no spin, true last CTA only)
        # for grids larger than the SM count.
        old = gl.atomic_add(cnt_ptr + t * 32, 1, sem="acq_rel", scope="gpu")
        phase = old % NSPLIT
        if phase >= NSPLIT - MERGE_PAR:
            idx_t = idx_ptr + t * stride_idx_t
            if row_multi:  # solo rows already wrote out (prefetched at top)
                target = old - phase + NSPLIT
                cur = old + 1
                while cur < target:
                    cur = gl.atomic_add(cnt_ptr + t * 32, 0, sem="acq_rel",
                                        scope="gpu")

                if MERGE_PAR == NSPLIT:
                    # Loop-free merge: ONE masked 3D load pulls every split's
                    # [H, DCQ] tile with the split axis in-thread
                    # (size_per_thread[0] = NSPLIT), so the axis-0 reductions
                    # are pure in-thread FMAs (no shuffles, no smem) and all
                    # NSPLIT tiles are one L2 burst. Register cost is
                    # NSPLIT*H*DCQ/128 = H*DC/128 = 64 regs for any NSPLIT
                    # since NSPLIT*DCQ == DC.
                    DCQ: gl.constexpr = DC // NSPLIT
                    L3D: gl.constexpr = gl.BlockedLayout(
                        [NSPLIT, 1, 2], [1, 2, 16], [1, 4, 1], [2, 1, 0])
                    LSH: gl.constexpr = gl.SliceLayout(2, L3D)   # [NS, H]
                    LHD: gl.constexpr = gl.SliceLayout(0, L3D)   # [H, DCQ]
                    hh = gl.arange(0, H, layout=gl.SliceLayout(0, LSH))
                    sp = gl.arange(0, NSPLIT, layout=gl.SliceLayout(1, LSH))
                    dq3 = phase * DCQ + gl.arange(
                        0, DCQ, layout=gl.SliceLayout(0, gl.SliceLayout(1, L3D)))
                    hh_o = gl.arange(0, H, layout=gl.SliceLayout(1, LHD))
                    dq_o = phase * DCQ + gl.arange(
                        0, DCQ, layout=gl.SliceLayout(0, LHD))
                    pml_b = pml_ptr + t * NSPLIT * 2 * H
                    pacc_b = pacc_ptr + t * NSPLIT * H * DC

                    alive = gl.load(idx_t + sp * CHUNK) >= 0           # [NS]
                    m_s = gl.load(pml_b + sp[:, None] * 2 * H + hh[None, :],
                                  mask=alive[:, None], other=float("-inf"))
                    l_s = gl.load(pml_b + sp[:, None] * 2 * H + H + hh[None, :],
                                  mask=alive[:, None], other=0.0)     # [NS,H]
                    a3 = gl.load(pacc_b + sp[:, None, None] * (H * DC)
                                 + hh[None, :, None] * DC + dq3[None, None, :],
                                 mask=alive[:, None, None], other=0.0)

                    m_g = gl.max(m_s, 0)                               # [H]
                    m_gs = gl.where(m_g == float("-inf"), 0.0, m_g)
                    w = gl.exp2(m_s - m_gs[None, :])                  # [NS,H]
                    l_g = gl.sum(l_s * w, 0)                           # [H]
                    l_safe = gl.where(l_g == 0.0, 1.0, l_g)
                    acc3 = gl.sum(a3 * w[:, :, None], 0)          # [H,DCQ]

                    l_safe_o = gl.convert_layout(l_safe,
                                                 gl.SliceLayout(1, LHD))
                    o3 = acc3 / l_safe_o[:, None]
                    gl.store(out_ptr + t * stride_o_t
                             + hh_o[:, None] * stride_o_h + dq_o[None, :],
                             o3.to(out_ptr.dtype.element_ty))
                    if phase == NSPLIT - 1:
                        lse3 = gl.where(l_g == 0.0, float("-inf"),
                                        m_g + gl.log2(l_safe))
                        gl.store(lse_ptr + t * stride_l_t + hh, lse3)
                else:
                    CL: gl.constexpr = gl.BlockedLayout([1, 4], [2, 16], [4, 1],
                                                        [1, 0])
                    CH: gl.constexpr = gl.SliceLayout(1, CL)
                    DCQ2: gl.constexpr = DC // MERGE_PAR
                    qslice = phase - (NSPLIT - MERGE_PAR)
                    hh = gl.arange(0, H, layout=CH)
                    dq = qslice * DCQ2 + gl.arange(0, DCQ2,
                                                   layout=gl.SliceLayout(0, CL))
                    pml_b = pml_ptr + t * NSPLIT * 2 * H
                    pacc_b = pacc_ptr + t * NSPLIT * H * DC

                    m_g = gl.full([H], float("-inf"), gl.float32, CH)
                    for s2 in gl.static_range(NSPLIT):
                        al = gl.load(idx_t + s2 * CHUNK) >= 0
                        m_row = gl.load(pml_b + s2 * 2 * H + hh,
                                        mask=al & (hh >= 0),
                                        other=float("-inf"))
                        m_g = gl.maximum(m_g, m_row)
                    m_gs = gl.where(m_g == float("-inf"), 0.0, m_g)

                    # Merge in groups of 4 splits: issue the 4 independent
                    # tile loads before any consumption so their DRAM/L2
                    # round-trips overlap.
                    l_g = gl.zeros([H], gl.float32, CH)
                    acc2 = gl.zeros([H, DCQ2], gl.float32, CL)
                    if NSPLIT < 4:  # generality fallback (T > 64 regimes)
                        for s2 in gl.static_range(NSPLIT):
                            al = gl.load(idx_t + s2 * CHUNK) >= 0
                            m_row = gl.load(pml_b + s2 * 2 * H + hh,
                                            mask=al & (hh >= 0),
                                            other=float("-inf"))
                            w = gl.exp2(m_row - m_gs)
                            l_row = gl.load(pml_b + s2 * 2 * H + H + hh,
                                            mask=al & (hh >= 0), other=0.0)
                            l_g += l_row * w
                            a2s = gl.load(pacc_b + s2 * H * DC
                                          + hh[:, None] * DC + dq[None, :],
                                          mask=al & (hh[:, None] >= 0),
                                          other=0.0)
                            acc2 += a2s * w[:, None]
                    for g in gl.static_range(0, NSPLIT // 4 * 4, 4):
                        al0 = gl.load(idx_t + (g + 0) * CHUNK) >= 0
                        al1 = gl.load(idx_t + (g + 1) * CHUNK) >= 0
                        al2 = gl.load(idx_t + (g + 2) * CHUNK) >= 0
                        al3 = gl.load(idx_t + (g + 3) * CHUNK) >= 0
                        m0 = gl.load(pml_b + (g + 0) * 2 * H + hh,
                                     mask=al0 & (hh >= 0), other=float("-inf"))
                        m1 = gl.load(pml_b + (g + 1) * 2 * H + hh,
                                     mask=al1 & (hh >= 0), other=float("-inf"))
                        m2 = gl.load(pml_b + (g + 2) * 2 * H + hh,
                                     mask=al2 & (hh >= 0), other=float("-inf"))
                        m3 = gl.load(pml_b + (g + 3) * 2 * H + hh,
                                     mask=al3 & (hh >= 0), other=float("-inf"))
                        l0 = gl.load(pml_b + (g + 0) * 2 * H + H + hh,
                                     mask=al0 & (hh >= 0), other=0.0)
                        l1 = gl.load(pml_b + (g + 1) * 2 * H + H + hh,
                                     mask=al1 & (hh >= 0), other=0.0)
                        l2 = gl.load(pml_b + (g + 2) * 2 * H + H + hh,
                                     mask=al2 & (hh >= 0), other=0.0)
                        l3 = gl.load(pml_b + (g + 3) * 2 * H + H + hh,
                                     mask=al3 & (hh >= 0), other=0.0)
                        a0 = gl.load(pacc_b + (g + 0) * H * DC
                                     + hh[:, None] * DC + dq[None, :],
                                     mask=al0 & (hh[:, None] >= 0), other=0.0)
                        a1 = gl.load(pacc_b + (g + 1) * H * DC
                                     + hh[:, None] * DC + dq[None, :],
                                     mask=al1 & (hh[:, None] >= 0), other=0.0)
                        a2 = gl.load(pacc_b + (g + 2) * H * DC
                                     + hh[:, None] * DC + dq[None, :],
                                     mask=al2 & (hh[:, None] >= 0), other=0.0)
                        a3 = gl.load(pacc_b + (g + 3) * H * DC
                                     + hh[:, None] * DC + dq[None, :],
                                     mask=al3 & (hh[:, None] >= 0), other=0.0)
                        w0 = gl.exp2(m0 - m_gs)      # 0 for dead splits
                        w1 = gl.exp2(m1 - m_gs)
                        w2 = gl.exp2(m2 - m_gs)
                        w3 = gl.exp2(m3 - m_gs)
                        l_g += l0 * w0 + l1 * w1 + l2 * w2 + l3 * w3
                        acc2 += a0 * w0[:, None] + a1 * w1[:, None]
                        acc2 += a2 * w2[:, None] + a3 * w3[:, None]
                    l_safe = gl.where(l_g == 0.0, 1.0, l_g)

                    o2 = acc2 / l_safe[:, None]
                    gl.store(out_ptr + t * stride_o_t
                             + hh[:, None] * stride_o_h + dq[None, :],
                             o2.to(out_ptr.dtype.element_ty))
                    if qslice == MERGE_PAR - 1:
                        lse2 = gl.where(l_g == 0.0, float("-inf"),
                                        m_g + gl.log2(l_safe))
                        gl.store(lse_ptr + t * stride_l_t + hh, lse2)


def _pick_nsplit(num_tokens: int) -> int:
    # Target ~256 CTAs (B200: 148 SMs); split the 2048 indices when T alone
    # can't fill the machine. Powers of two only so CHUNK divides TOPK.
    target = 256
    nsplit = 1
    while nsplit < 16 and num_tokens * nsplit < target:
        nsplit *= 2
    return nsplit


def kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    topk = sparse_indices.shape[-1]

    if num_tokens == 0:
        return

    nsplit = _pick_nsplit(num_tokens)

    # Counters are padded to 128B (stride 32 int32) so per-token spin polls
    # do not contend on one cache line.
    pacc, pml, cnt = _workspace(num_tokens * nsplit * num_qo_heads * head_dim_ckv,
                                num_tokens * nsplit * 2 * num_qo_heads,
                                num_tokens * 32,
                                q_nope.device)

    grid = (num_tokens, nsplit)
    _dsa_decode_gluon[grid](
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
        output, lse,
        pacc, pml, cnt,
        sm_scale * LOG2E,
        q_nope.stride(0), q_nope.stride(1),
        q_pe.stride(0), q_pe.stride(1),
        sparse_indices.stride(0),
        output.stride(0), output.stride(1),
        lse.stride(0),
        NSPLIT=nsplit,
        H=num_qo_heads, DC=head_dim_ckv, DP=head_dim_kpe,
        TOPK=topk, BLOCK_N=64,
        # Spin-merge needs the whole grid co-resident (B200: 148 SMs).
        # All-CTAs merge (one DC/NSPLIT slice each, loop-free 3D path) when
        # co-resident; serial true-last-CTA merge otherwise.
        MERGE_PAR=nsplit if num_tokens * nsplit <= 144 else 1,
        num_warps=4,
    )
