import torch
import triton
import triton.language as tl

# Gluon (Blackwell tcgen05) — used by the warp-specialized prefill GEMM1
# (exp_27 g3ws8: −29.7% vs the tl.dot persistent kernel on T=14107).
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia import blackwell as gbw
from triton.experimental.gluon.language.nvidia.blackwell import (
    mbarrier,
    tma as gtma,
    allocate_tensor_memory,
    tcgen05_mma,
    TensorMemoryLayout,
    fence_async_shared,
)
from triton.experimental.gluon.language.nvidia.ampere import async_copy as acp

_NUM_SMS = None
_TMA_ALLOC_SET = False


def _num_sms():
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(
            torch.cuda.current_device()).multi_processor_count
    return _NUM_SMS


def _ensure_tma_allocator():
    # Device-side tl.make_tensor_descriptor stages descriptors through
    # triton's allocator hook; install once.
    global _TMA_ALLOC_SET
    if not _TMA_ALLOC_SET:
        triton.set_allocator(
            lambda size, align, stream: torch.empty(size, device="cuda", dtype=torch.int8)
        )
        _TMA_ALLOC_SET = True


@triton.jit
def _routing(
    logits_ptr, bias_ptr, le_ptr, w_ptr,
    T, expert_offset, rsf,
    TB: tl.constexpr,
):
    # DeepSeek no-aux routing, one program per TB tokens, entirely in registers:
    # sigmoid -> group top-2 sums -> top-4 groups -> masked top-8 (argmax
    # extraction) -> normalized combine weights from s (bias-free).
    pid = tl.program_id(0)
    toks = pid * TB + tl.arange(0, TB)
    tmask = toks < T
    cols = tl.arange(0, 256)
    logits = tl.load(logits_ptr + toks[:, None] * 256 + cols[None, :],
                     mask=tmask[:, None], other=0.0)
    bias = tl.load(bias_ptr + cols).to(tl.float32)
    s = tl.sigmoid(logits)
    swb = s + bias[None, :]

    # group scores: top-2 sum within each group of 32 (mask the max by index,
    # not value — exact ties must keep their duplicate like torch.topk(2))
    g = tl.reshape(swb, (TB, 8, 32))
    m1 = tl.max(g, axis=2)
    i1 = tl.argmax(g, axis=2)
    g2 = tl.where(tl.arange(0, 32)[None, None, :] == i1[:, :, None], float('-inf'), g)
    m2 = tl.max(g2, axis=2)
    gs = m1 + m2                                     # [TB, 8]

    # top-4 groups -> keep mask
    keep = tl.zeros((TB, 8), dtype=tl.int1)
    g8 = tl.arange(0, 8)
    for _ in range(4):
        gi = tl.argmax(gs, axis=1)
        pick = g8[None, :] == gi[:, None]
        keep = keep | pick
        gs = tl.where(pick, float('-inf'), gs)
    keep256 = tl.reshape(tl.broadcast_to(keep[:, :, None], (TB, 8, 32)), (TB, 256))
    x = tl.where(keep256, swb, float('-inf'))

    # global top-8 on masked s_wb; weights gathered from s
    k8 = tl.arange(0, 8)
    idxs = tl.zeros((TB, 8), dtype=tl.int32)
    svals = tl.zeros((TB, 8), dtype=tl.float32)
    wsum = tl.zeros((TB,), dtype=tl.float32)
    for k in range(8):
        ki = tl.argmax(x, axis=1)
        pickc = cols[None, :] == ki[:, None]
        sv = tl.sum(tl.where(pickc, s, 0.0), axis=1)
        x = tl.where(pickc, float('-inf'), x)
        wsum += sv
        kk = k8[None, :] == k
        idxs = tl.where(kk, ki[:, None].to(tl.int32), idxs)
        svals = tl.where(kk, sv[:, None], svals)
    w8 = svals / (wsum[:, None] + 1e-20) * rsf

    # local expert id (sentinel 32 = not on this rank)
    le = idxs - expert_offset
    le = tl.where((le >= 0) & (le < 32), le, 32)
    off2 = toks[:, None] * 8 + k8[None, :]
    tl.store(le_ptr + off2, le, mask=tmask[:, None])
    tl.store(w_ptr + off2, w8, mask=tmask[:, None])


@triton.jit
def _hist(
    le_ptr, cnt_ptr, N,
    BLOCK: tl.constexpr,
):
    # Per-block histogram of local-expert ids (sentinel 32 lands in bin 32,
    # which is never read), then one 33-wide atomic add into global counts.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    le = tl.load(le_ptr + offs, mask=offs < N, other=32)
    h = tl.histogram(le, 64)            # power-of-2 bins; 33..63 always zero
    b64 = tl.arange(0, 64)
    tl.atomic_add(cnt_ptr + b64, h, mask=b64 < 33)


@triton.jit
def _tables(
    cnt_ptr, segoff_ptr,
    bexp1_ptr, brow1_ptr, bexp2_ptr, brow2_ptr, nbtot_ptr, nbtot1_ptr,
    NB1, NB2, NC1,
    BM1: tl.constexpr, BM2: tl.constexpr, CHUNK: tl.constexpr,
    DUAL: tl.constexpr,
):
    # Parallel over chunk-programs (the [32]-wide cumsums are recomputed per
    # program — trivial); program 0 also writes segoff + tile-row totals.
    # Programs [0, NC1) fill table 1; [NC1, ...) fill table 2 when DUAL.
    pid = tl.program_id(0)
    e32 = tl.arange(0, 32)
    counts = tl.load(cnt_ptr + e32)
    nb1 = (counts + (BM1 - 1)) // BM1
    nbc1 = tl.cumsum(nb1, 0)
    prev1 = nbc1 - nb1

    if pid == 0:
        segoff = tl.cumsum(counts, 0) - counts
        tl.store(segoff_ptr + e32, segoff)
        if DUAL:
            tl.store(nbtot1_ptr, tl.sum(nb1, axis=0))   # GEMM1 persistent tile rows
            nb2a = (counts + (BM2 - 1)) // BM2
            tl.store(nbtot_ptr, tl.sum(nb2a, axis=0))
        if not DUAL:
            tl.store(nbtot_ptr, tl.sum(nb1, axis=0))    # GEMM2 shares BM1 tables

    if pid < NC1:
        bid = pid * CHUNK + tl.arange(0, CHUNK)
        bmask = bid < NB1
        bexp = tl.sum((bid[:, None] >= nbc1[None, :]).to(tl.int32), axis=1)
        prev_g = tl.sum(tl.where(e32[None, :] == bexp[:, None], prev1[None, :], 0), axis=1)
        brow = (bid - prev_g) * BM1
        tl.store(bexp1_ptr + bid, bexp, mask=bmask)
        tl.store(brow1_ptr + bid, brow, mask=bmask)
    elif DUAL:
        nb2 = (counts + (BM2 - 1)) // BM2
        nbc2 = tl.cumsum(nb2, 0)
        prev2 = nbc2 - nb2
        bid = (pid - NC1) * CHUNK + tl.arange(0, CHUNK)
        bmask = bid < NB2
        bexp = tl.sum((bid[:, None] >= nbc2[None, :]).to(tl.int32), axis=1)
        prev_g = tl.sum(tl.where(e32[None, :] == bexp[:, None], prev2[None, :], 0), axis=1)
        brow = (bid - prev_g) * BM2
        tl.store(bexp2_ptr + bid, bexp, mask=bmask)
        tl.store(brow2_ptr + bid, brow, mask=bmask)


@triton.jit
def _scatter(
    le_ptr, w_ptr, segoff_ptr, cur_ptr, tok_ptr, wflat_ptr, inv_ptr,
    N,
    BLOCK: tl.constexpr,
):
    # Counting-sort scatter: each valid hit claims its slot in its expert's
    # segment via an atomic cursor; records token id, combine weight and the
    # flat->sorted inverse permutation used by _combine.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid0 = offs < N
    le = tl.load(le_ptr + offs, mask=valid0, other=32)
    w = tl.load(w_ptr + offs, mask=valid0, other=0.0)
    valid = valid0 & (le < 32)
    seg = tl.load(segoff_ptr + le, mask=valid, other=0)
    rank = tl.atomic_add(cur_ptr + le, 1, mask=valid)
    pos = seg + rank
    tl.store(tok_ptr + pos, (offs // 8).to(tl.int32), mask=valid)
    tl.store(wflat_ptr + pos, w, mask=valid)
    tl.store(inv_ptr + offs, pos, mask=valid)


@triton.jit
def _gemm1_tiny_splitk(
    A_ptr, As_ptr, W_ptr, Ws_ptr, P1_ptr, P2_ptr, le_ptr,
    T,
    BN: tl.constexpr, BK: tl.constexpr, H: tl.constexpr, I: tl.constexpr,
    KS: tl.constexpr,
):
    # Split-K over the 56-step k-chain: tiny-T is program-count-limited
    # (profile_tiny.md: 32 live programs at T=1, 800 GB/s of a 2.5 TB/s
    # floor). KS partials accumulate into fp32 staging; SwiGLU is applied
    # by _swiglu_tiny after the nonlinearity's inputs are complete.
    pid_h = tl.program_id(0)
    pid_n = tl.program_id(1)
    ks = tl.program_id(2)
    le = tl.load(le_ptr + pid_h)
    if le >= 32:
        return
    tok = pid_h // 8
    n0 = pid_n * BN
    offs_n = n0 + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    KSTEPS: tl.constexpr = H // BK // KS
    kb0 = ks * KSTEPS

    a_ptrs = A_ptr + tok * H + kb0 * BK + offs_k
    w_base = W_ptr + le.to(tl.int64) * (2 * I * H)
    b1_ptrs = w_base + offs_n[None, :].to(tl.int64) * H + kb0 * BK + offs_k[:, None]
    b2_ptrs = w_base + (offs_n[None, :] + I).to(tl.int64) * H + kb0 * BK + offs_k[:, None]
    ws_base = Ws_ptr + le * (32 * 56)
    nb1 = n0 // 128
    nb2 = (n0 + I) // 128

    acc1 = tl.zeros((16, BN), dtype=tl.float32)
    acc2 = tl.zeros((16, BN), dtype=tl.float32)
    for kb in range(kb0, kb0 + KSTEPS):
        a = tl.load(a_ptrs)
        a16 = tl.broadcast_to(a[None, :], (16, BK))
        asc = tl.load(As_ptr + kb * T + tok)
        b1 = tl.load(b1_ptrs)
        b2 = tl.load(b2_ptrs)
        s1 = tl.load(ws_base + nb1 * 56 + kb)
        s2 = tl.load(ws_base + nb2 * 56 + kb)
        acc1 += tl.dot(a16, b1) * (asc * s1)
        acc2 += tl.dot(a16, b2) * (asc * s2)
        a_ptrs += BK
        b1_ptrs += BK
        b2_ptrs += BK

    r16 = tl.arange(0, 16)
    v1 = tl.sum(tl.where(r16[:, None] == 0, acc1, 0.0), axis=0)
    v2 = tl.sum(tl.where(r16[:, None] == 0, acc2, 0.0), axis=0)
    tl.atomic_add(P1_ptr + pid_h * I + offs_n, v1)
    tl.atomic_add(P2_ptr + pid_h * I + offs_n, v2)


@triton.jit
def _swiglu_tiny(
    P1_ptr, P2_ptr, C_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    a1 = tl.load(P1_ptr + offs, mask=m, other=0.0)
    a2 = tl.load(P2_ptr + offs, mask=m, other=0.0)
    c = a1 * (a2 * tl.sigmoid(a2)) * 0.00390625
    tl.store(C_ptr + offs, c.to(tl.float16), mask=m)


@triton.jit
def _gemm1_tiny(
    A_ptr, As_ptr, W_ptr, Ws_ptr, C_ptr, le_ptr,
    T,
    BN: tl.constexpr, BK: tl.constexpr, H: tl.constexpr, I: tl.constexpr,
):
    # Per-(hit, n-tile) GEMV for tiny T: one token row, both SwiGLU halves,
    # no block tables, C indexed by flat hit id. fp32 vector math (M=1 —
    # tensor cores irrelevant, weight streaming dominates).
    pid_h = tl.program_id(0)
    pid_n = tl.program_id(1)
    le = tl.load(le_ptr + pid_h)
    if le >= 32:
        return
    tok = pid_h // 8
    n0 = pid_n * BN
    offs_n = n0 + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    a_ptrs = A_ptr + tok * H + offs_k
    w_base = W_ptr + le.to(tl.int64) * (2 * I * H)
    b1_ptrs = w_base + offs_n[None, :].to(tl.int64) * H + offs_k[:, None]
    b2_ptrs = w_base + (offs_n[None, :] + I).to(tl.int64) * H + offs_k[:, None]
    ws_base = Ws_ptr + le * (32 * 56)
    nb1 = n0 // 128
    nb2 = (n0 + I) // 128

    # Broadcast the single token row to tl.dot's minimum M=16 — tensor cores
    # at 8 KB acc beat an ALU reduction that spills 64 KB product tiles.
    acc1 = tl.zeros((16, BN), dtype=tl.float32)
    acc2 = tl.zeros((16, BN), dtype=tl.float32)
    for kb in range(0, H // BK):
        a = tl.load(a_ptrs)
        a16 = tl.broadcast_to(a[None, :], (16, BK))
        asc = tl.load(As_ptr + kb * T + tok)
        b1 = tl.load(b1_ptrs)
        b2 = tl.load(b2_ptrs)
        s1 = tl.load(ws_base + nb1 * 56 + kb)
        s2 = tl.load(ws_base + nb2 * 56 + kb)
        acc1 += tl.dot(a16, b1) * (asc * s1)
        acc2 += tl.dot(a16, b2) * (asc * s2)
        a_ptrs += BK
        b1_ptrs += BK
        b2_ptrs += BK

    c = acc1 * (acc2 * tl.sigmoid(acc2)) * 0.00390625
    r16 = tl.arange(0, 16)
    c_ptrs = C_ptr + pid_h * I + r16[:, None] * 0 + offs_n[None, :]
    tl.store(c_ptrs, c.to(tl.float16), mask=r16[:, None] == 0)


@triton.jit
def _gemm2_tiny(
    C_ptr, W_ptr, Ws_ptr, D_ptr, le_ptr, w8_ptr,
    BN: tl.constexpr, BK: tl.constexpr, H: tl.constexpr, I: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_n = tl.program_id(1)
    le = tl.load(le_ptr + pid_h)
    if le >= 32:
        return
    n0 = pid_n * BN
    offs_n = n0 + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    a_ptrs = C_ptr + pid_h * I + offs_k
    w_base = W_ptr + le.to(tl.int64) * (H * I)
    b_ptrs = w_base + offs_n[None, :].to(tl.int64) * I + offs_k[:, None]
    ws_base = Ws_ptr + le * (56 * 16)
    hb = n0 // 128

    acc = tl.zeros((16, BN), dtype=tl.float32)
    for kb in range(0, I // BK):
        a = tl.load(a_ptrs)
        a16 = tl.broadcast_to(a[None, :], (16, BK))
        b = tl.load(b_ptrs).to(tl.float16)
        s = tl.load(ws_base + hb * 16 + kb)
        acc += tl.dot(a16, b) * s
        a_ptrs += BK
        b_ptrs += BK

    wgt = tl.load(w8_ptr + pid_h)
    r16 = tl.arange(0, 16)
    d_ptrs = D_ptr + pid_h * H + r16[:, None] * 0 + offs_n[None, :]
    tl.store(d_ptrs, (acc * wgt).to(tl.float16), mask=r16[:, None] == 0)


@triton.jit
def _gemm1_swiglu_persistent(
    A_ptr, As_ptr, W_ptr, Ws_ptr, C_ptr,
    bexp_ptr, brow_ptr, counts_ptr, segoff_ptr, tok_ptr,
    T, nbtot_ptr, wrk_ptr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    H: tl.constexpr, I: tl.constexpr,
):
    # Prefill-only persistent CTAs with atomic work-stealing over real tiles.
    # Kept as a SEPARATE function: wrapping the k-loop in a while degrades
    # Triton's pipelining for the plain launch-per-tile variant (+3% decode).
    # Weight tiles stream via TMA descriptor (regular boxes over the
    # [E*2I, H] view): 517 -> 624 TFLOPS (experiments/tma_probe.md).
    w_desc = tl.make_tensor_descriptor(
        W_ptr, shape=[32 * 2 * I, H], strides=[H, 1], block_shape=[BN, BK],
    )
    nb_total = tl.load(nbtot_ptr)
    num_tiles = nb_total * (I // BN)
    tile = tl.atomic_add(wrk_ptr, 1)
    while tile < num_tiles:
        pid_m = tile // (I // BN)
        pid_n = tile % (I // BN)
        e = tl.load(bexp_ptr + pid_m)
        cnt = tl.load(counts_ptr + e)
        row0 = tl.load(brow_ptr + pid_m)
        seg = tl.load(segoff_ptr + e)
        rows = row0 + tl.arange(0, BM)
        m_mask = rows < cnt
        p = seg + rows
        tok = tl.load(tok_ptr + p, mask=m_mask, other=0)

        n0 = pid_n * BN
        offs_n = n0 + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)

        a_ptrs = A_ptr + tok[:, None].to(tl.int64) * H + offs_k[None, :]
        wrow1 = e * (2 * I) + n0
        wrow2 = wrow1 + I

        ws_base = Ws_ptr + e * (32 * 56)
        nb1 = n0 // 128
        nb2 = (n0 + I) // 128

        acc1 = tl.zeros((BM, BN), dtype=tl.float32)
        acc2 = tl.zeros((BM, BN), dtype=tl.float32)
        for kb in range(0, H // BK):
            a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
            b1 = w_desc.load([wrow1, kb * BK])           # [BN, BK]
            b2 = w_desc.load([wrow2, kb * BK])
            asc = tl.load(As_ptr + kb * T + tok, mask=m_mask, other=0.0)
            s1 = tl.load(ws_base + nb1 * 56 + kb)
            s2 = tl.load(ws_base + nb2 * 56 + kb)
            acc1 += tl.dot(a, tl.trans(b1)) * (asc * s1)[:, None]
            acc2 += tl.dot(a, tl.trans(b2)) * (asc * s2)[:, None]
            a_ptrs += BK

        c = acc1 * (acc2 * tl.sigmoid(acc2)) * 0.00390625
        c_ptrs = C_ptr + p[:, None].to(tl.int64) * I + offs_n[None, :]
        tl.store(c_ptrs, c.to(tl.float16), mask=m_mask[:, None])
        tile = tl.atomic_add(wrk_ptr, 1)

# ---------------------------------------------------------------------------
# Warp-specialized Gluon prefill GEMM1+SwiGLU (exp_27 "g3ws8").
# 12 warps: 8 consumer warps (default partition — MMA issue, per-128-block
# scale readback into register masters, SwiGLU epilogue) + 4 producer warps
# (worker partition, setmaxnreg 64 — cp.async A staging + TMA W ring, paced
# only by hardware-arrived MMA barriers so readback never back-pressures
# staging). Same math as _gemm1_swiglu_persistent: per-k-block
# acc += dot(a,w) * (a_scale[kb,row]*w_scale[kb]); fp16 C with 2^-8 prescale.
# mbarrier discipline: init ONCE; (NKB/NSTAGE) and NKB/2 even per tile so
# phases realign at tile boundaries (re-init of a live parity-1 mbarrier is
# NOT a reset — exp_26 deadlock lesson).
# Module-level constexprs: ws partition functions cannot take constexprs
# through warp_specialize args (triton 3.6).
# ---------------------------------------------------------------------------

_G3_BM = gl.constexpr(128)
_G38_BN = gl.constexpr(128)
_G3_BK = gl.constexpr(128)
_G3_H = gl.constexpr(7168)
_G3_I = gl.constexpr(2048)
_G3_NS = gl.constexpr(4)
_G3_NKB = gl.constexpr(56)


@gluon.jit
def _g3w8_producer(A_ptr, tok_ptr, seg, row0, cnt, wrow1, wrow2, w_desc,
                   a_smem, w1_smem, w2_smem, ld_bar, m1_bar):
    a_blk: gl.constexpr = gl.BlockedLayout([1, 16], [32, 1], [1, 4], [1, 0])
    offs_ka = gl.arange(0, _G3_BK, layout=gl.SliceLayout(0, a_blk))
    rows_a = row0 + gl.arange(0, _G3_BM, layout=gl.SliceLayout(1, a_blk))
    m_mask_a = rows_a < cnt
    tok_a = gl.load(tok_ptr + seg + rows_a, mask=m_mask_a, other=0)
    w_bytes: gl.constexpr = 2 * _G38_BN * _G3_BK
    for s in range(0, _G3_NS - 1):
        kcol = s * _G3_BK
        a_ptrs = A_ptr + tok_a[:, None].to(gl.int64) * _G3_H + (kcol + offs_ka)[None, :]
        acp.async_copy_global_to_shared(a_smem.index(s), a_ptrs,
                                        mask=m_mask_a[:, None])
        acp.commit_group()
        mbarrier.expect(ld_bar.index(s), w_bytes)
        gtma.async_copy_global_to_shared(w_desc, [wrow1, kcol], ld_bar.index(s), w1_smem.index(s))
        gtma.async_copy_global_to_shared(w_desc, [wrow2, kcol], ld_bar.index(s), w2_smem.index(s))
    for kb in range(0, _G3_NKB):
        nxt = kb + _G3_NS - 1
        acp.wait_group(_G3_NS - 2)
        fence_async_shared()
        if kb > 0:
            mbarrier.wait(m1_bar.index((kb - 1) & 1), ((kb - 1) >> 1) & 1)
        if nxt < _G3_NKB:
            nslot = nxt % _G3_NS
            kcoln = nxt * _G3_BK
            a_ptrs = A_ptr + tok_a[:, None].to(gl.int64) * _G3_H + (kcoln + offs_ka)[None, :]
            acp.async_copy_global_to_shared(a_smem.index(nslot), a_ptrs,
                                            mask=m_mask_a[:, None])
            mbarrier.expect(ld_bar.index(nslot), w_bytes)
            gtma.async_copy_global_to_shared(w_desc, [wrow1, kcoln], ld_bar.index(nslot), w1_smem.index(nslot))
            gtma.async_copy_global_to_shared(w_desc, [wrow2, kcoln], ld_bar.index(nslot), w2_smem.index(nslot))
        acp.commit_group()


@gluon.jit
def _g3w8_consumer_rb(As_ptr, C_ptr, tok_ptr, T, seg, row0, cnt, n0,
                      ws_base, nb1, nb2, a_smem, w1_smem, w2_smem,
                      acc1_t, acc2_t, ld_bar, m1_bar):
    acc_lay: gl.constexpr = TensorMemoryLayout([_G3_BM, _G38_BN], col_stride=1)
    reg_lay: gl.constexpr = gbw.get_tmem_reg_layout(
        gl.float32, [_G3_BM, _G38_BN], acc_lay, 8, "32x32b")
    row_lay: gl.constexpr = gl.SliceLayout(1, reg_lay)
    rows = row0 + gl.arange(0, _G3_BM, layout=row_lay)
    m_mask = rows < cnt
    p = seg + rows
    tok = gl.load(tok_ptr + p, mask=m_mask, other=0)
    acc1 = gl.zeros([_G3_BM, _G38_BN], gl.float32, layout=reg_lay)
    acc2 = gl.zeros([_G3_BM, _G38_BN], gl.float32, layout=reg_lay)
    asc_p = gl.zeros([_G3_BM], gl.float32, layout=row_lay)
    s1_p = 0.0
    s2_p = 0.0
    for kb in range(0, _G3_NKB):
        slot = kb % _G3_NS
        mbarrier.wait(ld_bar.index(slot), (kb // _G3_NS) & 1)
        fence_async_shared()
        pp = kb & 1
        tcgen05_mma(a_smem.index(slot), w1_smem.index(slot).permute((1, 0)),
                    acc1_t.index(pp), use_acc=False, mbarriers=[m1_bar.index(pp)])
        tcgen05_mma(a_smem.index(slot), w2_smem.index(slot).permute((1, 0)),
                    acc2_t.index(pp), use_acc=False, mbarriers=[m1_bar.index(pp)])
        if kb > 0:
            prev_pp = (kb - 1) & 1
            mph = ((kb - 1) >> 1) & 1
            mbarrier.wait(m1_bar.index(prev_pp), mph)
            acc1 += acc1_t.index(prev_pp).load(reg_lay) * (asc_p * s1_p)[:, None]
            acc2 += acc2_t.index(prev_pp).load(reg_lay) * (asc_p * s2_p)[:, None]
        asc_p = gl.load(As_ptr + kb * T + tok, mask=m_mask, other=0.0)
        s1_p = gl.load(ws_base + nb1 * 56 + kb)
        s2_p = gl.load(ws_base + nb2 * 56 + kb)
    prev_pp = (_G3_NKB - 1) & 1
    mph = ((_G3_NKB - 1) >> 1) & 1
    mbarrier.wait(m1_bar.index(prev_pp), mph)
    acc1 += acc1_t.index(prev_pp).load(reg_lay) * (asc_p * s1_p)[:, None]
    acc2 += acc2_t.index(prev_pp).load(reg_lay) * (asc_p * s2_p)[:, None]
    sig2 = 1.0 / (1.0 + gl.exp(-acc2))
    c = acc1 * (acc2 * sig2) * 0.00390625
    offs_n = n0 + gl.arange(0, _G38_BN, layout=gl.SliceLayout(0, reg_lay))
    c_ptrs = C_ptr + p[:, None].to(gl.int64) * _G3_I + offs_n[None, :]
    gl.store(c_ptrs, c.to(gl.float16), mask=m_mask[:, None])


@gluon.jit
def _g3ws8_gluon(
    A_ptr, As_ptr, W_ptr, Ws_ptr, C_ptr,
    bexp_ptr, brow_ptr, counts_ptr, segoff_ptr, tok_ptr,
    T, nbtot_ptr, wrk_ptr,
    BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
    H: gl.constexpr, I: gl.constexpr,
    NUM_WARPS: gl.constexpr, NSTAGE: gl.constexpr,
    PROD_REGS: gl.constexpr,
):
    gl.static_assert(BM == _G3_BM)
    gl.static_assert(BN == _G38_BN)
    gl.static_assert(BK == _G3_BK)
    gl.static_assert(H == _G3_H)
    gl.static_assert(I == _G3_I)
    gl.static_assert(NUM_WARPS == 8)
    gl.static_assert(NSTAGE == _G3_NS)
    NKB: gl.constexpr = H // BK
    gl.static_assert(NKB == _G3_NKB)
    gl.static_assert(NKB % NSTAGE == 0)
    gl.static_assert((NKB // NSTAGE) % 2 == 0)

    a_smem_lay: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BM, BK], gl.float8e4nv)
    w_smem_lay: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BN, BK], gl.float8e4nv)
    acc_lay: gl.constexpr = TensorMemoryLayout([BM, BN], col_stride=1)
    w_desc = gtma.make_tensor_descriptor(
        W_ptr, shape=[32 * 2 * I, H], strides=[H, 1], block_shape=[BN, BK], layout=w_smem_lay)
    a_smem = gl.allocate_shared_memory(gl.float8e4nv, [NSTAGE, BM, BK], a_smem_lay)
    w1_smem = gl.allocate_shared_memory(gl.float8e4nv, [NSTAGE, BN, BK], w_smem_lay)
    w2_smem = gl.allocate_shared_memory(gl.float8e4nv, [NSTAGE, BN, BK], w_smem_lay)
    acc1_t = allocate_tensor_memory(gl.float32, [2, BM, BN], acc_lay)
    acc2_t = allocate_tensor_memory(gl.float32, [2, BM, BN], acc_lay)
    ld_bar = gl.allocate_shared_memory(gl.int64, [NSTAGE, 1], mbarrier.MBarrierLayout())
    m1_bar = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    for i in range(NSTAGE):
        mbarrier.init(ld_bar.index(i), count=1)
    for i in range(2):
        mbarrier.init(m1_bar.index(i), count=2)

    nb_total = gl.load(nbtot_ptr)
    num_tiles = nb_total * (I // BN)
    tile = gl.atomic_add(wrk_ptr, 1, sem="relaxed")
    while tile < num_tiles:
        pid_m = tile // (I // BN)
        pid_n = tile % (I // BN)
        e = gl.load(bexp_ptr + pid_m)
        cnt = gl.load(counts_ptr + e)
        row0 = gl.load(brow_ptr + pid_m)
        seg = gl.load(segoff_ptr + e)
        n0 = pid_n * BN
        wrow1 = e * (2 * I) + n0
        wrow2 = wrow1 + I
        ws_base = Ws_ptr + e * (32 * 56)
        nb1 = n0 // 128
        nb2 = (n0 + I) // 128
        gl.warp_specialize(
            [(_g3w8_consumer_rb, (As_ptr, C_ptr, tok_ptr, T, seg, row0,
                                  cnt, n0, ws_base, nb1, nb2, a_smem,
                                  w1_smem, w2_smem, acc1_t, acc2_t,
                                  ld_bar, m1_bar)),
             (_g3w8_producer, (A_ptr, tok_ptr, seg, row0, cnt, wrow1,
                               wrow2, w_desc, a_smem, w1_smem, w2_smem,
                               ld_bar, m1_bar))],
            [4], [PROD_REGS])
        tile = gl.atomic_add(wrk_ptr, 1, sem="relaxed")


@triton.jit
def _gemm1_swiglu(
    A_ptr, As_ptr, W_ptr, Ws_ptr, C_ptr,
    bexp_ptr, brow_ptr, counts_ptr, segoff_ptr, tok_ptr,
    T,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    H: tl.constexpr, I: tl.constexpr,
):
    # One program: BM rows of one expert's token segment x BN cols of the
    # intermediate. Computes both SwiGLU halves (cols n and n+I of W13) so the
    # activation never round-trips at 2I width.
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    e = tl.load(bexp_ptr + pid_m)
    if e >= 32:
        return
    cnt = tl.load(counts_ptr + e)
    row0 = tl.load(brow_ptr + pid_m)
    if row0 >= cnt:
        return
    seg = tl.load(segoff_ptr + e)
    rows = row0 + tl.arange(0, BM)
    m_mask = rows < cnt
    p = seg + rows
    tok = tl.load(tok_ptr + p, mask=m_mask, other=0)

    n0 = pid_n * BN
    offs_n = n0 + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    a_ptrs = A_ptr + tok[:, None].to(tl.int64) * H + offs_k[None, :]
    w_base = W_ptr + e.to(tl.int64) * (2 * I * H)
    b1_ptrs = w_base + offs_n[None, :].to(tl.int64) * H + offs_k[:, None]
    b2_ptrs = w_base + (offs_n[None, :] + I).to(tl.int64) * H + offs_k[:, None]

    ws_base = Ws_ptr + e * (32 * 56)
    nb1 = n0 // 128
    nb2 = (n0 + I) // 128

    acc1 = tl.zeros((BM, BN), dtype=tl.float32)
    acc2 = tl.zeros((BM, BN), dtype=tl.float32)
    for kb in range(0, H // BK):
        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        b1 = tl.load(b1_ptrs)
        b2 = tl.load(b2_ptrs)
        asc = tl.load(As_ptr + kb * T + tok, mask=m_mask, other=0.0)
        s1 = tl.load(ws_base + nb1 * 56 + kb)
        s2 = tl.load(ws_base + nb2 * 56 + kb)
        acc1 += tl.dot(a, b1) * (asc * s1)[:, None]
        acc2 += tl.dot(a, b2) * (asc * s2)[:, None]
        a_ptrs += BK
        b1_ptrs += BK
        b2_ptrs += BK

    # SwiGLU: silu(x2) * x1, x1 = first half, x2 = second half.
    # Store as fp16 prescaled by 2^-8 (exact) so big activations can't
    # overflow fp16; GEMM2 folds the 2^8 back into the combine weight.
    c = acc1 * (acc2 * tl.sigmoid(acc2)) * 0.00390625
    c_ptrs = C_ptr + p[:, None].to(tl.int64) * I + offs_n[None, :]
    tl.store(c_ptrs, c.to(tl.float16), mask=m_mask[:, None])


@triton.jit
def _gemm2_grouped(
    C_ptr, W_ptr, Ws_ptr, D_ptr,
    bexp_ptr, brow_ptr, counts_ptr, segoff_ptr, wgt_ptr,
    nbtot_ptr, NUM_SMS,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    H: tl.constexpr, I: tl.constexpr,
    USE_TMA: tl.constexpr,
):
    # Persistent CTAs: grid = #SMs; each CTA strides over the REAL tile range
    # (nb_total row-tiles x H/BN col-tiles) — no dead-block overdispatch.
    # Weight tiles via TMA descriptor over the [E*H, I] view (exp_14 pattern);
    # gated off for the BW-bound decode band where trans/desc overhead nets +1-2%.
    if USE_TMA:
        w_desc = tl.make_tensor_descriptor(
            W_ptr, shape=[32 * H, I], strides=[I, 1], block_shape=[BN, BK],
        )
    pid = tl.program_id(0)
    nb_total = tl.load(nbtot_ptr)
    num_tiles = nb_total * (H // BN)
    tile = pid
    while tile < num_tiles:
        pid_m = tile // (H // BN)
        pid_n = tile % (H // BN)
        e = tl.load(bexp_ptr + pid_m)
        cnt = tl.load(counts_ptr + e)
        row0 = tl.load(brow_ptr + pid_m)
        seg = tl.load(segoff_ptr + e)
        rows = row0 + tl.arange(0, BM)
        m_mask = rows < cnt
        p = seg + rows
        wgt = tl.load(wgt_ptr + p, mask=m_mask, other=0.0)

        n0 = pid_n * BN
        offs_n = n0 + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)

        a_ptrs = C_ptr + p[:, None].to(tl.int64) * I + offs_k[None, :]
        wrow = e * H + n0
        b_ptrs = W_ptr + (e.to(tl.int64) * H + offs_n[None, :]) * I + offs_k[:, None]
        ws_base = Ws_ptr + e * (56 * 16)
        hb = n0 // 128

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for kb in range(0, I // BK):
            a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
            if USE_TMA:
                b = tl.trans(w_desc.load([wrow, kb * BK])).to(tl.float16)
            else:
                b = tl.load(b_ptrs).to(tl.float16)
            s = tl.load(ws_base + hb * 16 + kb)
            acc += tl.dot(a, b) * s
            a_ptrs += BK
            b_ptrs += BK

        # Per-hit contribution x combine weight, still carrying the 2^-8 C
        # prescale (kept for fp16 range safety; _combine multiplies it back).
        acc = acc * wgt[:, None]
        d_ptrs = D_ptr + p[:, None].to(tl.int64) * H + offs_n[None, :]
        tl.store(d_ptrs, acc.to(tl.float16), mask=m_mask[:, None])
        tile += NUM_SMS


@triton.jit
def _combine(
    D_ptr, le_ptr, inv_ptr, out_ptr,
    T,
    TB: tl.constexpr, BN: tl.constexpr, H: tl.constexpr,
    IDENT: tl.constexpr,
):
    # Sum each token's <=8 expert contributions (rows of D found via the
    # inverse sort permutation) and write bf16 output directly.
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    toks = pid_t * TB + tl.arange(0, TB)
    tmask = toks < T
    offs_n = pid_n * BN + tl.arange(0, BN)

    acc = tl.zeros((TB, BN), dtype=tl.float32)
    for k in range(8):
        le = tl.load(le_ptr + toks * 8 + k, mask=tmask, other=32)
        if IDENT:
            pk = toks * 8 + k          # tiny path: D is indexed by flat hit id
        else:
            pk = tl.load(inv_ptr + toks * 8 + k, mask=tmask, other=0)
        valid = tmask & (le < 32)
        d = tl.load(D_ptr + pk[:, None].to(tl.int64) * H + offs_n[None, :],
                    mask=valid[:, None], other=0.0)
        acc += d.to(tl.float32)

    out = (acc * 256.0).to(tl.bfloat16)
    o_ptrs = out_ptr + toks[:, None].to(tl.int64) * H + offs_n[None, :]
    tl.store(o_ptrs, out, mask=tmask[:, None])


@torch.no_grad()
def kernel(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
    output: torch.Tensor,
):
    H = 7168
    I = 2048
    T = routing_logits.shape[0]
    dev = hidden_states.device

    # ---- DeepSeek no-aux routing: one Triton kernel ----
    le_s = torch.empty((T, 8), dtype=torch.int32, device=dev)
    w8 = torch.empty((T, 8), dtype=torch.float32, device=dev)
    _routing[(triton.cdiv(T, 8),)](
        routing_logits, routing_bias, le_s, w8,
        T, local_expert_offset, routed_scaling_factor,
        TB=8, num_warps=4,
    )

    # ---- tiny-T fast path: per-hit GEMV, no sort/tables (hits ~= distinct
    # experts at this scale, so weight re-reads are ~1x) ----
    if T <= 16:
        C = torch.empty((8 * T, I), dtype=torch.float16, device=dev)
        D = torch.empty((8 * T, H), dtype=torch.float16, device=dev)
        le_flat = le_s.view(-1)
        if T <= 8:
            # Split-K only where live programs are scarce (profile_tiny.md:
            # 32 programs at T=1; by T~15 the band is already 1.6 waves)
            P = torch.zeros((2, 8 * T, I), dtype=torch.float32, device=dev)
            _gemm1_tiny_splitk[(8 * T, I // 128, 4)](
                hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale,
                P[0], P[1], le_flat, T, BN=128, BK=128, H=H, I=I, KS=4,
                num_warps=4, num_stages=4,
            )
            _swiglu_tiny[(triton.cdiv(8 * T * I, 4096),)](
                P[0], P[1], C, 8 * T * I, BLOCK=4096,
                num_warps=4,
            )
        else:
            _gemm1_tiny[(8 * T, I // 128)](
                hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale,
                C, le_flat, T, BN=128, BK=128, H=H, I=I,
                num_warps=4, num_stages=4,
            )
        _gemm2_tiny[(8 * T, H // 128)](
            C, gemm2_weights, gemm2_weights_scale, D, le_flat, w8.view(-1),
            BN=128, BK=128, H=H, I=I,
            num_warps=4, num_stages=4,
        )
        _combine[(triton.cdiv(T, 8), H // 256)](
            D, le_s, le_s, output,                                 # inv unused (IDENT)
            T, TB=8, BN=256, H=H, IDENT=True,
            num_warps=8,
        )
        return

    # ---- grouped (sorted-by-expert) layout via Triton counting sort ----
    # Configs by regime (empirical: tile_sweep.md + tile_sweep_decode.md —
    # decode segments are 1-6 rows, BM=16 doubles+ occupancy there):
    #   T >= 4096        : G1 (128, Gluon g3ws8 warp-spec) | G2 BM=128, ns=4, occ1, dual tables
    #   512 <= T < 4096  : G1 (64,ns4)            | G2 BM=64,  ns=3, occ2
    #   16 < T < 512     : G1 (16,ns3)            | G2 BM=16,  ns=4, occ2
    decode = T < 512
    dual = T >= 4096
    BM1 = 16 if decode else (128 if dual else 64)                  # GEMM1 row tile (g3ws8 needs 128)
    g1_ns = 3 if decode else 4
    BM2 = 128 if dual else BM1
    g2_ns = 3 if (512 <= T < 4096) else 4
    g2_occ = 1 if dual else 2
    NB1 = (8 * T + BM1 - 1) // BM1 + 32                            # worst-case block counts
    NB2 = (8 * T + BM2 - 1) // BM2 + 32
    cnt_cur = torch.zeros(68, dtype=torch.int32, device=dev)       # [0:33]=counts, [33:65]=cursors,
                                                                   # [65]/[66]=GEMM2/GEMM1 tile rows,
                                                                   # [67]=GEMM1 work counter
    segoff32 = torch.empty(32, dtype=torch.int32, device=dev)
    bexp1 = torch.empty(NB1, dtype=torch.int32, device=dev)
    brow1 = torch.empty(NB1, dtype=torch.int32, device=dev)
    if dual:
        bexp2 = torch.empty(NB2, dtype=torch.int32, device=dev)
        brow2 = torch.empty(NB2, dtype=torch.int32, device=dev)
    else:
        bexp2, brow2 = bexp1, brow1
    tok32 = torch.empty(8 * T, dtype=torch.int32, device=dev)
    wflat = torch.empty(8 * T, dtype=torch.float32, device=dev)
    inv = torch.empty(8 * T, dtype=torch.int32, device=dev)
    counts32 = cnt_cur[:32]

    _ensure_tma_allocator()                        # GEMM2 (all bands) + GEMM1 (dual) use TMA
    _hist[(triton.cdiv(8 * T, 2048),)](le_s, cnt_cur, 8 * T, BLOCK=2048, num_warps=8)
    NC1 = (NB1 + 255) // 256
    NC2 = (NB2 + 255) // 256 if dual else 0
    _tables[(NC1 + NC2,)](cnt_cur, segoff32, bexp1, brow1, bexp2, brow2,
                  cnt_cur[65:], cnt_cur[66:], NB1, NB2, NC1,
                  BM1=BM1, BM2=BM2, CHUNK=256, num_warps=4, DUAL=dual)
    _scatter[(triton.cdiv(8 * T, 1024),)](
        le_s, w8, segoff32, cnt_cur[33:65], tok32, wflat, inv, 8 * T,
        BLOCK=1024, num_warps=8,
    )

    C = torch.empty((8 * T, I), dtype=torch.float16, device=dev)
    D = torch.empty((8 * T, H), dtype=torch.float16, device=dev)

    # Persistent work-stealing only where tiles/CTA is high enough to
    # amortize (prefill); decode keeps the HW-scheduled per-tile grid.
    if dual:
        _ensure_tma_allocator()
        # exp_27 g3ws8: warp-specialized Gluon GEMM1 (−29.7% vs the tl.dot
        # persistent kernel at T=14107; tables built at BM1=128 above).
        _g3ws8_gluon[(_num_sms(),)](
            hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale, C,
            bexp1, brow1, counts32, segoff32, tok32,
            T, cnt_cur[66:], cnt_cur[67:],
            BM=128, BN=128, BK=128, H=H, I=I,
            NUM_WARPS=8, NSTAGE=4, PROD_REGS=64,
            num_warps=8,
        )
    else:
        _gemm1_swiglu[(I // 128, NB1)](
            hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale, C,
            bexp1, brow1, counts32, segoff32, tok32,
            T, BM=BM1, BN=128, BK=128, H=H, I=I,
            num_warps=8, num_stages=g1_ns,
        )
    # Persistent grid sized to the band's occupancy (smem budget per config).
    n_progs = _num_sms() * g2_occ
    _gemm2_grouped[(n_progs,)](
        C, gemm2_weights, gemm2_weights_scale, D,
        bexp2, brow2, counts32, segoff32, wflat,
        cnt_cur[65:], n_progs,
        BM=BM2, BN=128, BK=128, H=H, I=I, USE_TMA=not decode,
        num_warps=8, num_stages=g2_ns,
    )
    _combine[(triton.cdiv(T, 8), H // 256)](
        D, le_s, inv, output,
        T, TB=8, BN=256, H=H, IDENT=False,
        num_warps=8,
    )
