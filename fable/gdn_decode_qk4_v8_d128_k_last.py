import math

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl


@triton.jit
def _gdn_decode_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr,
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    out_ptr, new_state_ptr,
    scale,
    sq_b, sq_h, sq_k,
    sk_b, sk_h, sk_k,
    sv_b, sv_h, sv_v,
    ss_b, ss_h, ss_v, ss_k,
    sa_b, sa_h,
    sb_b, sb_h,
    so_b, so_h, so_v,
    sns_b, sns_h, sns_v, sns_k,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    HAS_STATE: tl.constexpr,
    STATE_FIRST: tl.constexpr,
):
    pid = tl.program_id(0)
    num_vb: tl.constexpr = V // BLOCK_V
    pid_b = pid // (H * num_vb)
    rem = pid % (H * num_vb)
    pid_h = rem // num_vb
    pid_v = rem % num_vb

    # Contiguity hints guarantee widest-vector LDG/STG on the K axis (exp_23).
    offs_k = tl.max_contiguous(tl.multiple_of(tl.arange(0, K), K), K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    qk_h = pid_h // 2  # GVA: v-head h uses q/k head h//2

    # State tile load issued FIRST: the dominant LDGs dispatch before the
    # gate/vector aux chain, which then overlaps the memory wait. (The B=8
    # tier regresses with this order — see exp_12 — hence the constexpr.)
    if HAS_STATE:
        s_ptrs = (state_ptr + pid_b * ss_b + pid_h * ss_h
                  + offs_v[:, None] * ss_v + offs_k[None, :] * ss_k)
        if STATE_FIRST:
            s = tl.load(s_ptrs, eviction_policy="evict_last")

    # Per-(b,h) scalar gates: g = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b)
    A_log = tl.load(A_log_ptr + pid_h)
    a_val = tl.load(a_ptr + pid_b * sa_b + pid_h * sa_h).to(tl.float32)
    dt = tl.load(dt_bias_ptr + pid_h)
    b_val = tl.load(b_ptr + pid_b * sb_b + pid_h * sb_h).to(tl.float32)
    x = a_val + dt
    # softplus with torch's threshold=20 behavior
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    g = tl.exp(-tl.exp(A_log) * sp)
    beta = tl.sigmoid(b_val)

    k_vec = tl.load(k_ptr + pid_b * sk_b + qk_h * sk_h + offs_k * sk_k).to(tl.float32)
    q_vec = tl.load(q_ptr + pid_b * sq_b + qk_h * sq_h + offs_k * sq_k).to(tl.float32)
    v_vec = tl.load(v_ptr + pid_b * sv_b + pid_h * sv_h + offs_v * sv_v).to(tl.float32)

    if HAS_STATE:
        if not STATE_FIRST:
            s = tl.load(s_ptrs, eviction_policy="evict_last")
        old_v = g * tl.sum(s * k_vec[None, :], axis=1)
        delta = beta * (v_vec - old_v)
        s_new = s * g + delta[:, None] * k_vec[None, :]
    else:
        delta = beta * v_vec
        s_new = delta[:, None] * k_vec[None, :]

    # Issue the big store before the out-reduction: the STGs drain while the
    # warp-shuffle reduction tree runs (independent dataflow).
    ns_ptrs = (new_state_ptr + pid_b * sns_b + pid_h * sns_h
               + offs_v[:, None] * sns_v + offs_k[None, :] * sns_k)
    tl.store(ns_ptrs, s_new, cache_modifier=".cs")

    out = scale * tl.sum(s_new * q_vec[None, :], axis=1)
    tl.store(out_ptr + pid_b * so_b + pid_h * so_h + offs_v * so_v,
             out.to(out_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# Gluon path: explicit layouts (B>8 tier, BLOCK_V=64, 8 warps).
#
# Coalescing constraint (measured): per-thread contiguous K run must equal the
# 16B vector width (4 f32) with lanes abutting, i.e. threads_per_warp=[1,32],
# so each warp instruction covers one full 512B row. The reduction-friendly
# [1,32]/[8,4] layout (32-elem runs/thread) was ~2x slower at B>=16: 16B
# chunks at 128B stride use half of every sector and stores degenerate to
# read-modify-write.
# ---------------------------------------------------------------------------

# [4,8]/[2,16] (hoping for 256-bit vectors) was +15-26%: Blackwell LDG.256
# does not emit; it degenerates to 16B chunks at 32B stride.
# B>=5 tiers: [64,128] tile, w8.
_L64 = ttgl.BlockedLayout([8, 4], [1, 32], [8, 1], [1, 0])
# [32,128] tile, w8 — tested at B=4 (+7%) and B=8 (+2.6%); unused.
_L32 = ttgl.BlockedLayout([4, 4], [1, 32], [8, 1], [1, 0])
# B=4 tier: [16,128] tile, w16, one V-row per warp.
_L16 = ttgl.BlockedLayout([1, 4], [1, 32], [16, 1], [1, 0])
# B<=2 tier: [8,128] tile, one V-row per warp (the small-tile winner).
_L8 = ttgl.BlockedLayout([1, 4], [1, 32], [8, 1], [1, 0])


@gluon.jit
def _gdn_decode_gluon(
    q_ptr, k_ptr, v_ptr, state_ptr,
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    out_ptr, new_state_ptr,
    scale,
    HQ: ttgl.constexpr,
    H: ttgl.constexpr,
    K: ttgl.constexpr,
    V: ttgl.constexpr,
    BLOCK_V: ttgl.constexpr,
    LAYOUT: ttgl.constexpr,
    HAS_STATE: ttgl.constexpr,
    STATE_FIRST: ttgl.constexpr,
):
    pid = ttgl.program_id(0)
    num_vb: ttgl.constexpr = V // BLOCK_V
    pid_b = pid // (H * num_vb)
    rem = pid % (H * num_vb)
    pid_h = rem // num_vb
    pid_v = rem % num_vb
    qk_h = pid_h // 2  # GVA: v-head h uses q/k head h//2

    layout_k: ttgl.constexpr = ttgl.SliceLayout(0, LAYOUT)
    layout_v: ttgl.constexpr = ttgl.SliceLayout(1, LAYOUT)
    offs_k = ttgl.arange(0, K, layout=layout_k)
    offs_v = pid_v * BLOCK_V + ttgl.arange(0, BLOCK_V, layout=layout_v)

    # Hardcoded contiguous strides (wrapper guards): state/new_state [B,H,V,K].
    if HAS_STATE:
        s_ptrs = (state_ptr + pid_b * (H * V * K) + pid_h * (V * K)
                  + ttgl.expand_dims(offs_v, 1) * K + ttgl.expand_dims(offs_k, 0))
        if STATE_FIRST:
            s = ttgl.load(s_ptrs, eviction_policy="evict_last")

    # Issue ALL loads before any dependent math: gluon skips Triton's
    # reorder-instructions pass, so source order is IR order — hoisting the
    # k/q/v loads above the transcendental gate chain restores the
    # memory-level parallelism Triton's scheduler created.
    A_log = ttgl.load(A_log_ptr + pid_h)
    a_val = ttgl.load(a_ptr + pid_b * H + pid_h).to(ttgl.float32)
    dt = ttgl.load(dt_bias_ptr + pid_h)
    b_val = ttgl.load(b_ptr + pid_b * H + pid_h).to(ttgl.float32)

    k_vec = ttgl.load(k_ptr + pid_b * (HQ * K) + qk_h * K + offs_k).to(ttgl.float32)
    q_vec = ttgl.load(q_ptr + pid_b * (HQ * K) + qk_h * K + offs_k).to(ttgl.float32)
    v_vec = ttgl.load(v_ptr + pid_b * (H * V) + pid_h * V + offs_v).to(ttgl.float32)

    if HAS_STATE:
        if not STATE_FIRST:  # alternate order knob (tested at (32,w8); unused)
            s = ttgl.load(s_ptrs, eviction_policy="evict_last")

    # Per-(b,h) scalar gates (softplus with torch's threshold=20 behavior).
    x = a_val + dt
    sp = ttgl.where(x > 20.0, x, ttgl.log(1.0 + ttgl.exp(x)))
    g = ttgl.exp(-ttgl.exp(A_log) * sp)
    beta = 1.0 / (1.0 + ttgl.exp(-b_val))

    k_row = ttgl.expand_dims(k_vec, 0)
    if HAS_STATE:
        old_v = g * ttgl.sum(s * k_row, axis=1)
        delta = beta * (v_vec - old_v)
        s_new = s * g + ttgl.expand_dims(delta, 1) * k_row
    else:
        delta = beta * v_vec
        s_new = ttgl.expand_dims(delta, 1) * k_row

    # Issue the big store before the out-reduction (independent dataflow).
    ns_ptrs = (new_state_ptr + pid_b * (H * V * K) + pid_h * (V * K)
               + ttgl.expand_dims(offs_v, 1) * K + ttgl.expand_dims(offs_k, 0))
    ttgl.store(ns_ptrs, s_new, cache_modifier=".cs")

    out = scale * ttgl.sum(s_new * ttgl.expand_dims(q_vec, 0), axis=1)
    ttgl.store(out_ptr + pid_b * (H * V) + pid_h * V + offs_v,
               out.to(out_ptr.dtype.element_ty))


def _gluon_fast_path_ok(q, k, v, state, a, b, output, new_state, HQ, H, K, V):
    """All tensors must match the hardcoded contiguous-stride addressing."""
    if (HQ, H, K, V) != (4, 8, 128, 128):
        return False
    if state is None or state.stride() != (H * V * K, V * K, K, 1):
        return False
    if new_state.stride() != (H * V * K, V * K, K, 1):
        return False
    if q.stride(3) != 1 or q.stride(2) != K or q.stride(0) != HQ * K:
        return False
    if k.stride(3) != 1 or k.stride(2) != K or k.stride(0) != HQ * K:
        return False
    if v.stride(3) != 1 or v.stride(2) != V or v.stride(0) != H * V:
        return False
    if output.stride(3) != 1 or output.stride(2) != V or output.stride(0) != H * V:
        return False
    if a.stride(2) != 1 or a.stride(0) != H or b.stride(2) != 1 or b.stride(0) != H:
        return False
    return True


@torch.no_grad()
def kernel(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state=None):
    """Gated Delta Net decode, fused single-launch kernel (k-last state layout)."""
    B, T, num_q_heads, K = q.shape
    num_v_heads = v.shape[2]
    V = v.shape[3]

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    if new_state is None:
        new_state = torch.empty((B, num_v_heads, V, K), dtype=torch.float32, device=q.device)

    # Gluon path with explicit layouts + hand-hoisted loads (gluon skips
    # Triton's reorder-instructions pass, so the hoisting is done in source).
    # Paired A/B vs exp_23 Triton kernel, 27/27 workload wins (x2 runs):
    # B=1 -4.4%, B=4 -3.7%, B=8 -5.5%, B=16 -4.9%, B=32 -4.2%, B=48 -3.7%,
    # B=64 -5.5%. Note B=8 jumps to the (64,w8) tile (single wave, 128 CTAs):
    # -6.0% there vs +2.6..3.5% on (32,w8)/(16,w16).
    if _gluon_fast_path_ok(q, k, v, state, a, b, output, new_state,
                           num_q_heads, num_v_heads, K, V):
        if B <= 2:
            BLOCK_V, LAYOUT, W = 8, _L8, 8
        elif B <= 4:
            BLOCK_V, LAYOUT, W = 8, _L8, 8
        else:
            BLOCK_V, LAYOUT, W = 64, _L64, 8
        grid = (B * num_v_heads * (V // BLOCK_V),)
        _gdn_decode_gluon[grid](
            q, k, v, state,
            A_log, a, dt_bias, b,
            output, new_state,
            scale,
            HQ=num_q_heads,
            H=num_v_heads,
            K=K,
            V=V,
            BLOCK_V=BLOCK_V,
            LAYOUT=LAYOUT,
            HAS_STATE=True,
            STATE_FIRST=True,
            num_warps=W,
        )
        return output, new_state

    # B-adaptive V-tiling: small batches need more CTAs to fill the GPU's SMs
    # (B=1 with BLOCK_V=64 is only 16 CTAs on 148 SMs); large batches are
    # bandwidth-bound and prefer fewer, fatter tiles.
    if B <= 2:
        BLOCK_V, num_warps = 8, 8
    elif B <= 4:
        BLOCK_V, num_warps = 16, 16
    elif B <= 8:
        BLOCK_V, num_warps = 32, 8
    else:
        BLOCK_V, num_warps = 64, 8
    grid = (B * num_v_heads * (V // BLOCK_V),)

    has_state = state is not None
    state_arg = state if has_state else q  # dummy ptr, never dereferenced
    ss = state.stride() if has_state else (0, 0, 0, 0)

    _gdn_decode_kernel[grid](
        q, k, v, state_arg,
        A_log, a, dt_bias, b,
        output, new_state,
        scale,
        q.stride(0), q.stride(2), q.stride(3),
        k.stride(0), k.stride(2), k.stride(3),
        v.stride(0), v.stride(2), v.stride(3),
        ss[0], ss[1], ss[2], ss[3],
        a.stride(0), a.stride(2),
        b.stride(0), b.stride(2),
        output.stride(0), output.stride(2), output.stride(3),
        new_state.stride(0), new_state.stride(1), new_state.stride(2), new_state.stride(3),
        H=num_v_heads,
        K=K,
        V=V,
        BLOCK_V=BLOCK_V,
        HAS_STATE=has_state,
        STATE_FIRST=BLOCK_V != 32,  # B=8 tier prefers the original load order (exp_12)
        num_warps=num_warps,
    )
    return output, new_state
