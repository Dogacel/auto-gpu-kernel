"""
GDN prefill (gated delta rule), chunked implementation.

Hybrid: Triton `_gdn_wy` / `_gdn_out` (unchanged from solution_fused.py) +
a Gluon (tcgen05) inter-chunk state pass `_gdn_state_gluon`.

Why Gluon for the state pass: the recurrence
    U_c = u0_c - W_c @ S          (S = chunk-start state, [K,V])
    S   = exp(cg_last)*S + Kd_c^T @ U_c
is serial over chunks within one (seq, head). In Triton each tl.dot round-trips
the loop-carried state TMEM->registers->SMEM every chunk (4.2-7.15 us/iter
measured). Here the state lives in TMEM across the whole chain, transposed
(St = S^T, [BV,K]) so it is always the *A* operand of tcgen05 MMAs:

    MMA1: ut   = -u0^T + St @ W^T        (=> ut = -U^T)        acc preloaded
    MMA2: St'  = g*St  + ut @ (-Kd)      (= (g S + Kd^T U)^T)  acc preloaded

The decay scale g*St is a TMEM->reg->TMEM roundtrip into the *other* ping-pong
buffer, overlapped with MMA1 (both only read St). All loads / kd build / u0
preload for chunk c+1 are prefetched during chunk c's MMAs. The same TMEM
read used for the scale provides the hc checkpoint for free.

tf32 note: tcgen05 SMEM operands of 32-bit dtype must be K-major, hence W^T is
a permuted view of row-major w (k contiguous) and Kd is stored transposed
[K,BT] (t contiguous). Precision model identical to the Triton tf32 dots.
"""

import math

import torch
import triton
import triton.language as tl

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    get_tmem_reg_layout,
    mbarrier,
    tcgen05_mma,
    tma,
)


@triton.jit
def _softplus(x):
    # matches F.softplus(x) with threshold=20
    return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))


@triton.jit
def _tri_minv(Bm, o_i, levels):
    """X with Minv = I + X, for M = I + L where Bm = -L is strictly lower [BT,BT].

    Hierarchical block inversion (see solution_fused.py provenance)."""
    X = tl.where((o_i[:, None] // 2) == (o_i[None, :] // 2), Bm, 0.0)
    s = 2
    for _ in tl.range(0, levels):
        same2s = (o_i[:, None] // (2 * s)) == (o_i[None, :] // (2 * s))
        lowblk = same2s & ((o_i[:, None] // s - o_i[None, :] // s) == 1)
        Loff = -tl.where(lowblk, Bm, 0.0)
        # bf16 operands (f32 acc, f32 carry): 2x dot throughput on this
        # 10-dot latency chain; rounding enters per-multiply, no compounding
        P = Loff + tl.dot(X.to(tl.bfloat16), Loff.to(tl.bfloat16))
        P = P + tl.dot(P.to(tl.bfloat16), X.to(tl.bfloat16))
        X = X - P
        s = s * 2
    return X


@triton.jit
def _chunk_lookup(cu_seqlens_ptr, pid, N, BT: tl.constexpr, BN: tl.constexpr):
    """Map global chunk id -> (seq idx, global chunk base of that seq, total chunks)."""
    offs = tl.arange(0, BN)
    m = offs < N
    lo = tl.load(cu_seqlens_ptr + offs, mask=m, other=0)
    hi = tl.load(cu_seqlens_ptr + offs + 1, mask=m, other=0)
    nch = (hi - lo + BT - 1) // BT
    cum = tl.cumsum(nch, 0)
    total = tl.sum(nch, 0)
    n = tl.sum((cum <= pid).to(tl.int64), 0)
    base = tl.sum(tl.where(offs == n - 1, cum, 0), 0)
    return n, base, total


@triton.jit
def _gdn_wy(
    k_ptr, v_ptr, a_ptr, b_ptr, A_log_ptr, dt_bias_ptr, cu_seqlens_ptr,
    cg_ptr, w_ptr, u0_ptr, kdt_ptr, gamma_ptr,
    s_kt, s_kh, s_kd, s_vt, s_vh, s_vd,
    s_at, s_ah, s_bt, s_bh,
    s_wh, s_kdh,
    N, inv_steps,
    EMIT_KDT: tl.constexpr,
    HV: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BN: tl.constexpr,
):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    n, base, total = _chunk_lookup(cu_seqlens_ptr, pid, N, BT, BN)
    if pid >= total:
        return
    bos = tl.load(cu_seqlens_ptr + n)
    eos = tl.load(cu_seqlens_ptr + n + 1)
    lc = pid - base
    rows = bos + lc * BT + tl.arange(0, BT)
    valid = rows < eos

    A_log_h = tl.load(A_log_ptr + h)
    dt_h = tl.load(dt_bias_ptr + h)
    a_col = tl.load(a_ptr + rows * s_at + h * s_ah, mask=valid, other=0.0).to(tl.float32)
    b_col = tl.load(b_ptr + rows * s_bt + h * s_bh, mask=valid, other=0.0).to(tl.float32)
    lg = -tl.exp(A_log_h) * _softplus(a_col + dt_h)
    lg = tl.where(valid, lg, 0.0)
    cg = tl.cumsum(lg, 0)
    beta = tl.sigmoid(b_col)
    tl.store(cg_ptr + rows * HV + h, cg, mask=valid)

    hk = h // 2
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    kt = tl.load(k_ptr + rows[:, None] * s_kt + hk * s_kh + offs_k[None, :] * s_kd,
                 mask=valid[:, None], other=0.0)  # [BT,BK] bf16
    vt = tl.load(v_ptr + rows[:, None] * s_vt + h * s_vh + offs_v[None, :] * s_vd,
                 mask=valid[:, None], other=0.0)  # [BT,BV] bf16

    o_i = tl.arange(0, BT)
    kk = tl.dot(kt, tl.trans(kt))  # [BT,BT] f32
    decay = tl.exp(tl.minimum(cg[:, None] - cg[None, :], 0.0))
    Bm = -tl.where(o_i[:, None] > o_i[None, :], kk * decay * beta[:, None], 0.0)
    Bm = _tri_minv(Bm, o_i, inv_steps)  # now Minv = I + Bm

    bgk = (beta * tl.exp(cg))[:, None] * kt.to(tl.float32)  # [BT,BK]
    wt = bgk + tl.dot(Bm.to(tl.bfloat16), bgk.to(tl.bfloat16))
    bv = beta[:, None] * vt.to(tl.float32)  # [BT,BV]
    u0 = bv + tl.dot(Bm.to(tl.bfloat16), bv.to(tl.bfloat16))

    wu_base = (rows[:, None] * HV + h) * BK
    # w is stored HEAD-MAJOR [HV, T, BK] so the state pass can TMA 2D tiles
    tl.store(w_ptr + h * s_wh + rows[:, None] * BK + offs_k[None, :], wt,
             mask=valid[:, None])
    tl.store(u0_ptr + wu_base + offs_v[None, :], u0, mask=valid[:, None])

    # precompute for the gluon state pass: -Kd^T tile (head-major [HV, NTB*BK, BT])
    # and the chunk decay gamma. kd rows past eos are exact 0 (kt masked to 0).
    # T<=64 workloads route to _gdn_state_tr which recomputes kd from k+cg and
    # never reads these — gate the stores off there (exp_25).
    if EMIT_KDT:
        cg_last = tl.min(cg, 0)  # lg masked to 0 past eos -> cumsum flat -> min = last valid
        nkdt = -tl.trans(kt.to(tl.float32) * tl.exp(cg_last - cg)[:, None])  # [BK,BT]
        tl.store(kdt_ptr + h * s_kdh + (pid * BK + offs_k[:, None]) * BT + o_i[None, :], nkdt)
        tl.store(gamma_ptr + pid * HV + h, tl.exp(cg_last))


# ---------------------------------------------------------------------------
# Gluon state pass (warp specialized, 3 partitions)
#
# Default partition: ONLY tcgen05 MMAs + mbarrier waits. Any generic SMEM/TMEM
# access in the same instruction stream as the MMAs makes the compiler insert
# tcgen05.wait pipe-drains (measured 5 us/iter); a pure-MMA partition runs at
# the ~0.7 us/iter hardware floor of the dependent MMA pair.
# Scale partition (WW warps): state init, the decay-scale TMEM roundtrip
# (g*S into the other S ping-pong buffer), hc checkpoint store, final
# new_state store.
# IO partition (WW warps): u0^T global loads + TMEM preloads, vnew stores.
# Splitting scale from io overlaps their serial latencies (exp_19); the two
# worker partitions never touch the same TMEM buffer.
#
# Barriers (count=1, phase = chunk parity; arrive is always elected-single-
# thread AFTER a partition-local thread_barrier so it covers every warp):
#   b_tma:  TMA bytes for chunk c's w/kd slots; armed by mma
#   b_init: scale -> mma   "s_bufs[0] state init done" (one-shot, phase 0)
#   b_u0:   io    -> mma   "u0 preload for chunk c ready"
#   b_inb:  scale -> mma   "MMA2 acc preload (g*S) for chunk c ready"
#   b_u:    mma commit MMA1_c; waited by mma (before MMA2) and io (vnew)
#   b_s:    mma commit MMA2_c; waited by mma (next MMA1), scale, io
#   b_fb:   io    -> mma   "io consumed b_s_c"; MMA2_{c+1} waits it so
#           b_s_{c+1} cannot complete before io's parity check of b_s_c
#           (mod-2 alias overrun, the exp_17 deadlock class)
#
# Strict back-pressure invariant (re-derived for 3 partitions): for every
# barrier, completion #k+1 is transitively gated on EVERY #k-waiter having
# passed its wait. Producer gates: b_u0_{c+1} <= io passed b_u_c (so MMA1_c
# issued, so mma passed b_u0_c); b_inb_{c+1} <= scale passed b_s_c (so MMA2_c
# issued, so mma passed b_inb_c); b_u_{c+1} <= MMA1_{c+1} <= b_u0_{c+1} <= io
# passed b_u_c; b_s_{c+1} <= MMA2_{c+1} <= b_inb_{c+1} (scale passed b_s_c)
# AND b_fb_c (io passed b_s_c) AND mma program order; b_fb_{c+1} <= io passed
# b_s_{c+1} <= MMA2_{c+1} <= mma passed b_fb_c.
# ---------------------------------------------------------------------------

@gluon.jit
def _state_mma_part(w_desc, kdt_desc, s_bufs, u_bufs, w_slots, kdt_slots,
                    b_tma, b_init, b_u0, b_inb, b_u, b_s, b_fb, bos, nch, cbase,
                    BT: gl.constexpr, BK: gl.constexpr):
    # prologue: TMA chunk-0 operands
    mbarrier.expect(b_tma, 2 * BT * BK * 4)
    tma.async_copy_global_to_shared(w_desc, [bos, 0], b_tma, w_slots.index(0))
    tma.async_copy_global_to_shared(kdt_desc, [cbase * BK, 0], b_tma,
                                    kdt_slots.index(0))
    # MMA1_0 reads s_bufs[0], initialized by the scale partition (b_u0 no
    # longer covers it: it is arrived by the io partition)
    mbarrier.wait(b_init, phase=0)
    phase = 0
    for c in range(nch):
        cur = c % 2
        nxt = (c + 1) % 2
        mbarrier.wait(b_tma, phase=phase)
        mbarrier.wait(b_u0, phase=phase)
        mbarrier.wait(b_s, phase=phase ^ 1, pred=c > 0)
        # arm + issue TMA for chunk c+1 — only now: the nxt slots' previous
        # readers (MMA1/MMA2 of chunk c-1) are complete per b_u/b_s
        more = c + 1 < nch
        mbarrier.expect(b_tma, 2 * BT * BK * 4, pred=more)
        tma.async_copy_global_to_shared(w_desc, [bos + (c + 1) * BT, 0], b_tma,
                                        w_slots.index(nxt), pred=more)
        tma.async_copy_global_to_shared(kdt_desc, [(cbase + c + 1) * BK, 0], b_tma,
                                        kdt_slots.index(nxt), pred=more)
        # MMA1: ut = -u0^T + St @ W^T
        tcgen05_mma(s_bufs.index(cur), w_slots.index(cur).permute((1, 0)),
                    u_bufs.index(cur), use_acc=True, mbarriers=[b_u])
        mbarrier.wait(b_inb, phase=phase)
        mbarrier.wait(b_u, phase=phase)
        # io's parity check of b_s_{c-1} must precede b_s_c's completion
        mbarrier.wait(b_fb, phase=phase ^ 1, pred=c > 0)
        # MMA2: St' = g*St + ut @ (-Kd)
        tcgen05_mma(u_bufs.index(cur), kdt_slots.index(cur).permute((1, 0)),
                    s_bufs.index(nxt), use_acc=True, mbarriers=[b_s])
        phase = phase ^ 1


@gluon.jit
def _state_scale_part(
    state_ptr, gamma_ptr, hc_ptr, new_state_ptr,
    s_bufs, b_init, b_inb, b_s,
    nch, cbase, n, h, vb,
    s_sn, s_sh, s_sv, s_nn, s_nh, s_nv,
    HAS_STATE: gl.constexpr, HV: gl.constexpr,
    BK: gl.constexpr, BV: gl.constexpr, WW: gl.constexpr,
):
    s_tl: gl.constexpr = TensorMemoryLayout((BV, BK), col_stride=1)
    s_reg: gl.constexpr = get_tmem_reg_layout(gl.float32, (BV, BK), s_tl, WW)
    offs_v = gl.arange(0, BV, layout=gl.SliceLayout(1, s_reg))
    offs_k = gl.arange(0, BK, layout=gl.SliceLayout(0, s_reg))
    offs_vg = vb * BV + offs_v

    # ---- prologue: initial state ----
    if HAS_STATE:
        st0 = gl.load(state_ptr + n * s_sn + h * s_sh + offs_vg[:, None] * s_sv
                      + offs_k[None, :])
    else:
        st0 = gl.zeros((BV, BK), gl.float32, layout=s_reg)
    s_bufs.index(0).store(st0)
    gl.thread_barrier()
    mbarrier.arrive(b_init, count=1)

    gamma = gl.load(gamma_ptr + cbase * HV + h)
    phase = 0
    for c in range(nch):
        cur = c % 2
        nxt = (c + 1) % 2
        # issue chunk-(c+1) gamma load early (independent of the MMA chain)
        cn = gl.minimum(c + 1, nch - 1)
        gamma_n = gl.load(gamma_ptr + (cbase + cn) * HV + h)

        # scale (needs MMA2_{c-1} complete)
        mbarrier.wait(b_s, phase=phase ^ 1, pred=c > 0)
        st_prev = s_bufs.index(cur).load(s_reg)
        s_bufs.index(nxt).store(st_prev * gamma)
        gl.thread_barrier()
        mbarrier.arrive(b_inb, count=1)

        # checkpoint store (off the chain; same TMEM read as the scale)
        cid = cbase + c
        gl.store(hc_ptr + ((cid * HV + h) * BK + offs_k[None, :]) * BK + offs_vg[:, None],
                 st_prev)

        phase = phase ^ 1
        gamma = gamma_n

    mbarrier.wait(b_s, phase=(nch - 1) % 2)
    sf = s_bufs.index(nch % 2).load(s_reg)
    gl.store(new_state_ptr + n * s_nn + h * s_nh + offs_vg[:, None] * s_nv
             + offs_k[None, :], sf)


@gluon.jit
def _state_io_part(
    u0_ptr, vnew_ptr,
    u_bufs, b_u0, b_u, b_s, b_fb,
    bos, eos, nch, h, vb,
    HV: gl.constexpr, BT: gl.constexpr,
    BK: gl.constexpr, BV: gl.constexpr, WW: gl.constexpr,
):
    u_tl: gl.constexpr = TensorMemoryLayout((BV, BT), col_stride=1)
    u_reg: gl.constexpr = get_tmem_reg_layout(gl.float32, (BV, BT), u_tl, WW)
    offs_vu = vb * BV + gl.arange(0, BV, layout=gl.SliceLayout(1, u_reg))
    offs_tu = gl.arange(0, BT, layout=gl.SliceLayout(0, u_reg))

    # ---- prologue: chunk-0 u0 preload + chunk-1 prefetch ----
    rows_u = bos + offs_tu
    nu0t = -gl.load(u0_ptr + (rows_u * HV + h)[None, :] * BK + offs_vu[:, None],
                    mask=(rows_u < eos)[None, :], other=0.0)
    u_bufs.index(0).store(nu0t)
    gl.thread_barrier()
    mbarrier.arrive(b_u0, count=1)
    c1 = gl.minimum(1, nch - 1)
    rows_n = bos + c1 * BT + offs_tu
    nu0t = -gl.load(u0_ptr + (rows_n * HV + h)[None, :] * BK + offs_vu[:, None],
                    mask=(rows_n < eos)[None, :], other=0.0)

    phase = 0
    for c in range(nch):
        cur = c % 2
        nxt = (c + 1) % 2

        # u0 preload for c+1 (loaded one iteration ahead, so the global-load
        # latency stays off the b_u0 arrive path). u_bufs[nxt] is free: its
        # last reader MMA2_{c-1} completed per the b_s wait at the tail of
        # iteration c-1 (prologue-fresh for c=0).
        u_bufs.index(nxt).store(nu0t)
        cn = gl.minimum(c + 2, nch - 1)
        rows_n = bos + cn * BT + offs_tu
        nu0t = -gl.load(u0_ptr + (rows_n * HV + h)[None, :] * BK + offs_vu[:, None],
                        mask=(rows_n < eos)[None, :], other=0.0)

        # vnew (= U = -ut) for chunk c
        mbarrier.wait(b_u, phase=phase)
        # arrive b_u0 ("chunk c+1 u0 ready") only AFTER every io warp observed
        # b_u_c: b_u_{c+1} <= MMA1_{c+1} <= b_u0_{c+1}, so our b_u parity check
        # can never be overrun (exp_17 deadlock class).
        gl.thread_barrier()
        mbarrier.arrive(b_u0, count=1)
        u_val = u_bufs.index(cur).load(u_reg)
        rows_c = bos + c * BT + offs_tu
        gl.store(vnew_ptr + (rows_c * HV + h)[None, :] * BK + offs_vu[:, None], -u_val,
                 mask=(rows_c < eos)[None, :])

        # consume b_s_c, then feed back: MMA2_{c+1} waits b_fb_c, so b_s_{c+1}
        # cannot complete before every io warp passed this parity check.
        mbarrier.wait(b_s, phase=phase)
        gl.thread_barrier()
        mbarrier.arrive(b_fb, count=1)

        phase = phase ^ 1


@gluon.jit
def _gdn_state_gluon(
    state_ptr, cu_seqlens_ptr,
    gamma_ptr, w_ptr, u0_ptr, kdt_ptr, vnew_ptr, hc_ptr, new_state_ptr,
    s_sn, s_sh, s_sv,
    s_nn, s_nh, s_nv,
    s_wh, s_kdh, Tlen, NTBK,
    N,
    HAS_STATE: gl.constexpr,
    HV: gl.constexpr, BT: gl.constexpr, BK: gl.constexpr, BV: gl.constexpr,
    BN: gl.constexpr, WARPS: gl.constexpr,
):
    n = gl.program_id(0)
    h = gl.program_id(1)
    vb = gl.program_id(2)

    # cu-scan with a warp-replicated layout: the reduce lowers to shuffles only.
    # NO cross-warp (smem+bar.sync) reduction and NO thread_barrier may appear
    # in this parent region — the worker-partition warps are still parked at
    # the warp_specialize entry and a CTA-wide bar.sync would deadlock.
    SPT: gl.constexpr = (BN + 31) // 32
    lay1: gl.constexpr = gl.BlockedLayout([SPT], [32], [4], [0])
    offs_n = gl.arange(0, BN, layout=lay1)
    mN = offs_n < N
    lo = gl.load(cu_seqlens_ptr + offs_n, mask=mN, other=0)
    hi = gl.load(cu_seqlens_ptr + offs_n + 1, mask=mN, other=0)
    nch_all = (hi - lo + BT - 1) // BT
    cbase = gl.cast(gl.sum(gl.where(offs_n < n, nch_all, 0), 0), gl.int32)

    bos = gl.cast(gl.load(cu_seqlens_ptr + n), gl.int32)
    eos = gl.cast(gl.load(cu_seqlens_ptr + n + 1), gl.int32)
    nch = (eos - bos + BT - 1) // BT

    s_tl: gl.constexpr = TensorMemoryLayout((BV, BK), col_stride=1)
    u_tl: gl.constexpr = TensorMemoryLayout((BV, BT), col_stride=1)
    s_bufs = allocate_tensor_memory(gl.float32, (2, BV, BK), s_tl)
    u_bufs = allocate_tensor_memory(gl.float32, (2, BV, BT), u_tl)

    w_lay: gl.constexpr = gl.NVMMASharedLayout.get_default_for((BT, BK), gl.float32)
    kdt_lay: gl.constexpr = gl.NVMMASharedLayout.get_default_for((BK, BT), gl.float32)
    w_slots = gl.allocate_shared_memory(gl.float32, (2, BT, BK), w_lay)
    kdt_slots = gl.allocate_shared_memory(gl.float32, (2, BK, BT), kdt_lay)

    w_desc = tma.make_tensor_descriptor(
        w_ptr + h * s_wh, [Tlen, BK], [BK, 1], [BT, BK], w_lay)
    kdt_desc = tma.make_tensor_descriptor(
        kdt_ptr + h * s_kdh, [NTBK, BT], [BT, 1], [BK, BT], kdt_lay)

    b_tma = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    b_init = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    b_u0 = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    b_inb = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    b_u = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    b_s = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    b_fb = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(b_tma, count=1)
    mbarrier.init(b_init, count=1)
    mbarrier.init(b_u0, count=1)
    mbarrier.init(b_inb, count=1)
    mbarrier.init(b_u, count=1)
    mbarrier.init(b_s, count=1)
    mbarrier.init(b_fb, count=1)
    # no thread_barrier here: worker warps are parked until the fork; the
    # warp_specialize entry handshake orders these inits for them.

    gl.warp_specialize(
        [
            (_state_mma_part,
             (w_desc, kdt_desc, s_bufs, u_bufs, w_slots, kdt_slots,
              b_tma, b_init, b_u0, b_inb, b_u, b_s, b_fb, bos, nch, cbase,
              BT, BK)),
            (_state_scale_part,
             (state_ptr, gamma_ptr, hc_ptr, new_state_ptr,
              s_bufs, b_init, b_inb, b_s,
              nch, cbase, n, h, vb,
              s_sn, s_sh, s_sv, s_nn, s_nh, s_nv,
              HAS_STATE, HV, BK, BV, WARPS)),
            (_state_io_part,
             (u0_ptr, vnew_ptr,
              u_bufs, b_u0, b_u, b_s, b_fb,
              bos, eos, nch, h, vb,
              HV, BT, BK, BV, WARPS)),
        ],
        [WARPS, WARPS], [184, 184],
    )


@triton.jit
def _gdn_state_tr(
    k_ptr, state_ptr, cu_seqlens_ptr,
    cg_ptr, w_ptr, u0_ptr, vnew_ptr, hc_ptr, new_state_ptr,
    s_kt, s_kh, s_kd,
    s_sn, s_sh, s_sv, s_sk,
    s_nn, s_nh, s_nv, s_nk,
    s_wh,
    N,
    HAS_STATE: tl.constexpr,
    HV: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BN: tl.constexpr,
):
    """Plain Triton state pass (exp_13 lineage) for single-chunk workloads where
    the gluon pipeline's fixed costs (TMA allocator, barrier prologue) dominate.
    Reads the head-major w layout the gluon-era _gdn_wy emits."""
    n = tl.program_id(0)
    h = tl.program_id(1)
    vb = tl.program_id(2)

    offs_n = tl.arange(0, BN)
    m = offs_n < N
    lo = tl.load(cu_seqlens_ptr + offs_n, mask=m, other=0)
    hi = tl.load(cu_seqlens_ptr + offs_n + 1, mask=m, other=0)
    nch_all = (hi - lo + BT - 1) // BT
    cbase = tl.sum(tl.where(offs_n < n, nch_all, 0), 0)

    bos = tl.load(cu_seqlens_ptr + n)
    eos = tl.load(cu_seqlens_ptr + n + 1)
    nch = (eos - bos + BT - 1) // BT

    offs_k = tl.arange(0, BK)
    offs_v = vb * BV + tl.arange(0, BV)
    hk = h // 2

    if HAS_STATE:
        S = tl.load(state_ptr + n * s_sn + h * s_sh
                    + offs_v[None, :] * s_sv + offs_k[:, None] * s_sk).to(tl.float32)
    else:
        S = tl.zeros([BK, BV], dtype=tl.float32)

    for c in range(0, nch):
        cid = cbase + c
        rows = bos + c * BT + tl.arange(0, BT)
        valid = rows < eos
        hc_off = ((cid * HV + h) * BK + offs_k[:, None]) * BK + offs_v[None, :]
        tl.store(hc_ptr + hc_off, S)

        wu_base = (rows[:, None] * HV + h) * BK
        wt = tl.load(w_ptr + h * s_wh + rows[:, None] * BK + offs_k[None, :],
                     mask=valid[:, None], other=0.0)
        u0 = tl.load(u0_ptr + wu_base + offs_v[None, :], mask=valid[:, None], other=0.0)
        cg = tl.load(cg_ptr + rows * HV + h, mask=valid, other=float('inf'))
        kt = tl.load(k_ptr + rows[:, None] * s_kt + hk * s_kh + offs_k[None, :] * s_kd,
                     mask=valid[:, None], other=0.0)

        U = u0 - tl.dot(wt, S, input_precision="tf32")  # [BT,BV]
        tl.store(vnew_ptr + wu_base + offs_v[None, :], U, mask=valid[:, None])

        cg_last = tl.min(cg, 0)  # cg non-increasing; min over valid = last valid
        kd = kt.to(tl.float32) * tl.exp(cg_last - cg)[:, None]
        S = S * tl.exp(cg_last) + tl.dot(tl.trans(kd), U, input_precision="tf32")

    tl.store(new_state_ptr + n * s_nn + h * s_nh
             + offs_v[None, :] * s_nv + offs_k[:, None] * s_nk, S)


@triton.jit
def _gdn_out(
    q_ptr, k_ptr, cu_seqlens_ptr,
    cg_ptr, vnew_ptr, hc_ptr, out_ptr,
    s_qt, s_qh, s_qd, s_kt, s_kh, s_kd, s_ot, s_oh, s_od,
    scale, N,
    HV: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BN: tl.constexpr,
):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    vb = tl.program_id(2)
    n, base, total = _chunk_lookup(cu_seqlens_ptr, pid, N, BT, BN)
    if pid >= total:
        return
    bos = tl.load(cu_seqlens_ptr + n)
    eos = tl.load(cu_seqlens_ptr + n + 1)
    lc = pid - base
    rows = bos + lc * BT + tl.arange(0, BT)
    valid = rows < eos
    offs_k = tl.arange(0, BK)
    offs_v = vb * BV + tl.arange(0, BV)
    hk = h // 2

    qt = tl.load(q_ptr + rows[:, None] * s_qt + hk * s_qh + offs_k[None, :] * s_qd,
                 mask=valid[:, None], other=0.0)
    kt = tl.load(k_ptr + rows[:, None] * s_kt + hk * s_kh + offs_k[None, :] * s_kd,
                 mask=valid[:, None], other=0.0)
    cg = tl.load(cg_ptr + rows * HV + h, mask=valid, other=0.0)
    wu_base = (rows[:, None] * HV + h) * BK
    U = tl.load(vnew_ptr + wu_base + offs_v[None, :], mask=valid[:, None], other=0.0)
    pid64 = pid.to(tl.int64)
    hc_off = ((pid64 * HV + h) * BK + offs_k[:, None]) * BK + offs_v[None, :]
    Hc = tl.load(hc_ptr + hc_off)  # [BK,BV] f32

    o_i = tl.arange(0, BT)
    P = tl.dot(qt, tl.trans(kt))  # [BT,BT] f32
    P = P * tl.exp(tl.minimum(cg[:, None] - cg[None, :], 0.0))
    P = tl.where(o_i[:, None] >= o_i[None, :], P, 0.0)  # inclusive diagonal

    qf = qt.to(tl.float32) * tl.exp(cg)[:, None]
    # single-hop to bf16 output (inherent 0.4% quantization): bf16 operands,
    # f32 acc — 2x dot throughput, error enters once
    O = tl.dot(qf.to(tl.bfloat16), Hc.to(tl.bfloat16)) \
        + tl.dot(P.to(tl.bfloat16), U.to(tl.bfloat16))
    O = O * scale
    tl.store(out_ptr + rows[:, None] * s_ot + h * s_oh + offs_v[None, :] * s_od,
             O.to(out_ptr.dtype.element_ty), mask=valid[:, None])


_TMA_ALLOCATOR_SET = False


def _ensure_tma_allocator():
    """Device-side TMA descriptors need a global-scratch allocator."""
    global _TMA_ALLOCATOR_SET
    if not _TMA_ALLOCATOR_SET:
        def _alloc(size: int, alignment: int, stream):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(_alloc)
        _TMA_ALLOCATOR_SET = True


@torch.no_grad()
def kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state):
    T, HQ, D = q.shape
    HV = v.shape[1]
    N = cu_seqlens.numel() - 1

    assert HQ == 4 and HV == 8 and D == 128

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(D)

    BT = 64
    BK = D
    BV = 64
    BVS = 64  # gluon state pass V-block (M of the tcgen05 MMAs)
    WARPS = 4  # per worker partition (scale + io); 4 vs 8 measured equal pre-split

    NTB = T // BT + N + 1  # upper bound on total #chunks (programs beyond exit early)
    BN = triton.next_power_of_2(N + 1)
    dev = q.device

    cg_buf = torch.empty((T, HV), dtype=torch.float32, device=dev)
    w_buf = torch.empty((HV, T, BK), dtype=torch.float32, device=dev)  # head-major (TMA)
    u0_buf = torch.empty((T, HV, BK), dtype=torch.float32, device=dev)
    # exp_23: vnew/hc stored bf16 — _gdn_out casts both U and Hc to bf16 at its
    # dots (since exp_21), so the same RNE rounding moves from load-site to
    # store-site: dot operands bit-identical, outputs unchanged. Halves the hc
    # checkpoint traffic (0.26 us/iter pole, exp_22 decomposition) + vnew store
    # (~0.13 us/iter) + halves _gdn_out's Hc/U load bytes.
    vnew_buf = torch.empty((T, HV, BK), dtype=torch.bfloat16, device=dev)
    hc_buf = torch.empty((NTB, HV, BK, BK), dtype=torch.bfloat16, device=dev)
    kdt_buf = torch.empty((HV, NTB * BK, BT), dtype=torch.float32, device=dev)
    gamma_buf = torch.empty((NTB, HV), dtype=torch.float32, device=dev)

    _gdn_wy[(NTB, HV)](
        k, v, a, b, A_log, dt_bias, cu_seqlens,
        cg_buf, w_buf, u0_buf, kdt_buf, gamma_buf,
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        w_buf.stride(0), kdt_buf.stride(0),
        N, 5,
        EMIT_KDT=T > BT,
        HV=HV, BT=BT, BK=BK, BV=BK, BN=BN,
        num_warps=8,
    )

    if state is not None:
        assert state.stride(3) == 1
        st, st_strides = state, (state.stride(0), state.stride(1), state.stride(2))
    else:
        st, st_strides = q, (0, 0, 0)
    assert new_state.stride(3) == 1

    if T <= BT:
        # single-chunk workloads: gluon's fixed costs (TMA allocator, barrier
        # prologue) cost +5-11% here (exp_17 A/B, crossover ~0.040 ms / T<=64);
        # the plain Triton state pass wins below it
        _gdn_state_tr[(N, HV, D // 16)](
            k, st, cu_seqlens,
            cg_buf, w_buf, u0_buf, vnew_buf, hc_buf, new_state,
            k.stride(0), k.stride(1), k.stride(2),
            st_strides[0], st_strides[1], st_strides[2], 1,
            new_state.stride(0), new_state.stride(1), new_state.stride(2), 1,
            w_buf.stride(0),
            N,
            HAS_STATE=state is not None,
            HV=HV, BT=BT, BK=BK, BV=16, BN=BN,
            num_warps=8,
        )
        _gdn_out[(NTB, HV, D // BV)](
            q, k, cu_seqlens,
            cg_buf, vnew_buf, hc_buf, output,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            scale, N,
            HV=HV, BT=BT, BK=BK, BV=BV, BN=BN,
            num_warps=8,
        )
        return output, new_state

    _ensure_tma_allocator()
    _gdn_state_gluon[(N, HV, D // BVS)](
        st, cu_seqlens,
        gamma_buf, w_buf, u0_buf, kdt_buf, vnew_buf, hc_buf, new_state,
        st_strides[0], st_strides[1], st_strides[2],
        new_state.stride(0), new_state.stride(1), new_state.stride(2),
        w_buf.stride(0), kdt_buf.stride(0), T, NTB * BK,
        N,
        HAS_STATE=state is not None,
        HV=HV, BT=BT, BK=BK, BV=BVS, BN=BN, WARPS=WARPS,
        num_warps=4,  # default (MMA) partition; WARPS each for scale + io partitions
    )

    _gdn_out[(NTB, HV, D // BV)](
        q, k, cu_seqlens,
        cg_buf, vnew_buf, hc_buf, output,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        scale, N,
        HV=HV, BT=BT, BK=BK, BV=BV, BN=BN,
        num_warps=8,
    )

    return output, new_state
