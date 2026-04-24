"""
DSA TopK Indexer — exp 42: score_kernel ported to Gluon, other kernels unchanged.

Gluon is Triton's explicit-scheduling dialect (triton.experimental.gluon). The port
uses helpers from triton.tools.triton_to_gluon_translater.translator_helpers to wrap
the layout-aware variants of arange/full/trans/dot so the kernel body stays close to
the original Triton semantics.

Only `score_kernel` is ported. `fast_small_kernel`, `scoreless_kernel`,
`radix_topk_kernel`, and the `kernel()` wrapper are unchanged from exp 37.
"""

import torch
import triton
import triton.language as tl

# Gluon imports
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.tools.triton_to_gluon_translater.translator_helpers import (
    tl_dot,
    tl_arange,
    tl_full,
    tl_trans,
    default_blocked_layout,
    reset_to_default_layout,
)


@triton.jit
def fast_small_kernel(
    seq_lens_ptr, block_table_ptr, topk_indices_ptr,
    stride_btb,
    stride_out_b, stride_out_k,
    BLOCK_T: tl.constexpr,
    TOPK: tl.constexpr,
):
    # Exp 25: mp=1 fast path collapses to pure index arithmetic.
    pid_b = tl.program_id(0)

    seq_len = tl.load(seq_lens_ptr + pid_b)
    page_id = tl.load(block_table_ptr + pid_b * stride_btb).to(tl.int64)

    k_offs = tl.arange(0, TOPK)
    out_ptrs = topk_indices_ptr + pid_b * stride_out_b + k_offs * stride_out_k
    tl.store(out_ptrs, tl.full([TOPK], -1, tl.int32))

    actual_topk = tl.minimum(seq_len.to(tl.int32), BLOCK_T)
    t_offs_out = tl.arange(0, BLOCK_T)
    token_idx = (page_id * BLOCK_T + t_offs_out).to(tl.int32)
    final_idx = tl.where(t_offs_out < actual_topk, token_idx, -1)
    real_out_ptrs = topk_indices_ptr + pid_b * stride_out_b + t_offs_out * stride_out_k
    tl.store(real_out_ptrs, final_idx)


@triton.jit
def scoreless_kernel(
    seq_lens_ptr, block_table_ptr, topk_indices_ptr,
    stride_bt_b, stride_bt_p,
    stride_out_b, stride_out_k,
    page_size: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Exp 26: scoreless path for max_num_pages <= 32.
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)

    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_in_topk = k_offs < topk

    seq_len = tl.load(seq_lens_ptr + pid_b)
    actual_topk = tl.minimum(seq_len.to(tl.int32), topk)

    page_idx = k_offs // page_size
    offset = k_offs % page_size

    k_in_range = k_offs < actual_topk

    bt_ptrs = block_table_ptr + pid_b * stride_bt_b + page_idx * stride_bt_p
    global_page = tl.load(bt_ptrs, mask=k_in_range, other=0).to(tl.int64)

    token_idx = (global_page * page_size + offset).to(tl.int32)

    final = tl.where(k_in_range, token_idx, -1)

    out_ptrs = topk_indices_ptr + pid_b * stride_out_b + k_offs * stride_out_k
    tl.store(out_ptrs, final, mask=k_in_topk)


@gluon.jit
def score_kernel(
    q_ptr, k_fp8_ptr, k_scale_ptr, w_ptr,
    seq_lens_ptr, block_table_ptr, scores_ptr,
    stride_qb, stride_qh, stride_qd,
    stride_kp, stride_kt, stride_kd,
    stride_ksp, stride_kst,
    stride_wb, stride_wh,
    stride_btb, stride_btp,
    stride_sb, stride_st,
    BLOCK_H: ttgl.constexpr,
    BLOCK_D: ttgl.constexpr,
    BLOCK_T: ttgl.constexpr,
):
    # Gluon port of exp 37 score_kernel. Preserves 2D grid (batch_size, max_num_pages),
    # early-return, FP8 MMA, relu + weight multiply, cross-head sum, scalar scale-after-
    # sum, and -1e30 sentinel writes. Uses helpers from the triton_to_gluon translator
    # so layout plumbing (arange/full/trans/broadcast/dot) is handled library-side.

    pid_b = ttgl.program_id(0)
    pid_p = ttgl.program_id(1)

    seq_len = ttgl.load(seq_lens_ptr + pid_b)
    token_start = pid_p * BLOCK_T

    # Early-return for tiles fully beyond seq_len -- exp 9 optimization.
    if token_start >= seq_len:
        t_offs_sk = tl_arange(0, BLOCK_T)
        score_off_sk = pid_b * stride_sb + (token_start + t_offs_sk) * stride_st
        ttgl.store(scores_ptr + score_off_sk, tl_full([BLOCK_T], -1e30, ttgl.float32))
        return

    page_id_raw = ttgl.load(block_table_ptr + pid_b * stride_btb + pid_p * stride_btp)
    page_id = page_id_raw.to(ttgl.int64)

    h_offs = tl_arange(0, BLOCK_H)
    d_offs = tl_arange(0, BLOCK_D)
    t_offs = tl_arange(0, BLOCK_T)

    # 2D offset computation requires broadcasting the 1D aranges into 2D layouts.
    # The auto-translator wraps h_offs[:, None] * stride_qh + d_offs[None, :] * stride_qd
    # with SliceLayout conversions; here we match that pattern.
    h_layout_2d: ttgl.constexpr = ttgl.SliceLayout(
        1, default_blocked_layout([BLOCK_H, BLOCK_D], ttgl.num_warps())
    )
    d_layout_h2d: ttgl.constexpr = ttgl.SliceLayout(
        0, default_blocked_layout([BLOCK_H, BLOCK_D], ttgl.num_warps())
    )
    t_layout_2d: ttgl.constexpr = ttgl.SliceLayout(
        1, default_blocked_layout([BLOCK_T, BLOCK_D], ttgl.num_warps())
    )
    d_layout_t2d: ttgl.constexpr = ttgl.SliceLayout(
        0, default_blocked_layout([BLOCK_T, BLOCK_D], ttgl.num_warps())
    )

    q_off = (
        pid_b * stride_qb
        + ttgl.convert_layout(h_offs, h_layout_2d)[:, None] * stride_qh
        + ttgl.convert_layout(d_offs, d_layout_h2d)[None, :] * stride_qd
    )
    q_fp8 = ttgl.load(q_ptr + q_off)

    k_off = (
        page_id * stride_kp
        + ttgl.convert_layout(t_offs, t_layout_2d)[:, None] * stride_kt
        + ttgl.convert_layout(d_offs, d_layout_t2d)[None, :] * stride_kd
    )
    k_fp8 = ttgl.load(k_fp8_ptr + k_off)

    # Scalar loads issued early so they overlap with the MMA (exp 20 hoist).
    s_off = page_id * stride_ksp + t_offs * stride_kst
    scale = ttgl.load(k_scale_ptr + s_off)
    w_off = pid_b * stride_wb + h_offs * stride_wh
    w = ttgl.load(w_ptr + w_off)

    # FP8 tensor-core dot via the library Blackwell helper (tcgen05_mma when supported,
    # mma_v2 otherwise). `tl_trans` uses the reset_to_default_layout wrapper so the
    # resulting tensor has a normal blocked layout consumable by tl_dot.
    scores = tl_dot(q_fp8, reset_to_default_layout(tl_trans(k_fp8)), out_dtype=ttgl.float32)

    # Scale >= 0 commute (exp 10): relu + per-head weight broadcast + cross-head sum,
    # then scalar scale-per-t multiply after the sum.
    scores = ttgl.maximum(scores, 0.0)

    # scores: [BLOCK_H, BLOCK_T] -> multiply by w broadcast along T.
    w_layout_2d: ttgl.constexpr = ttgl.SliceLayout(
        1, default_blocked_layout([BLOCK_H, BLOCK_T], ttgl.num_warps())
    )
    w_2d = ttgl.convert_layout(w, w_layout_2d)
    scores = scores * w_2d[:, None]

    # sum over H -> [BLOCK_T]
    final = reset_to_default_layout(ttgl.sum(scores, axis=0)) * scale

    abs_t = token_start + t_offs
    in_bounds = abs_t < seq_len
    final = ttgl.where(in_bounds, final, -1e30)

    score_off = pid_b * stride_sb + abs_t * stride_st
    ttgl.store(scores_ptr + score_off, final)


@triton.jit
def radix_topk_kernel(
    scores_ptr,
    seq_lens_ptr,
    block_table_ptr,
    topk_indices_ptr,
    stride_sb, stride_sn,
    stride_btb, stride_btp,
    stride_out_b, stride_out_k,
    max_scored,
    max_num_pages,
    page_size: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Exp 28/29/33/37 state — unchanged.
    pid_b = tl.program_id(0)
    seq_len = tl.load(seq_lens_ptr + pid_b)

    out_k_offs = tl.arange(0, topk)

    if seq_len <= topk:
        actual_topk = tl.minimum(seq_len.to(tl.int32), topk)
        in_range_sc = out_k_offs < actual_topk
        page_idx_sc = out_k_offs // page_size
        offset_sc = out_k_offs % page_size
        bt_ptrs_sc = block_table_ptr + pid_b * stride_btb + page_idx_sc * stride_btp
        global_page_sc = tl.load(bt_ptrs_sc, mask=in_range_sc, other=0).to(tl.int64)
        token_idx_sc = (global_page_sc * page_size + offset_sc).to(tl.int32)
        final_sc = tl.where(in_range_sc, token_idx_sc, -1)
        out_ptrs_sc = topk_indices_ptr + pid_b * stride_out_b + out_k_offs * stride_out_k
        tl.store(out_ptrs_sc, final_sc)
        return

    offs = tl.arange(0, BLOCK_N)
    in_bounds = offs < max_scored
    score_ptrs = scores_ptr + pid_b * stride_sb + offs * stride_sn
    scores = tl.load(score_ptrs, mask=in_bounds, other=float('-inf'))

    page_idx = offs // page_size
    offset = offs % page_size
    page_idx_clamped = tl.minimum(page_idx, max_num_pages - 1)
    bt_ptrs = block_table_ptr + pid_b * stride_btb + page_idx_clamped * stride_btp
    global_page = tl.load(bt_ptrs, mask=in_bounds, other=0).to(tl.int64)
    token_idx = (global_page * page_size + offset).to(tl.int32)

    score_bits = scores.to(tl.uint32, bitcast=True)
    sign = score_bits >> 31
    xor_mask = tl.where(
        sign != 0,
        tl.full([BLOCK_N], 0xFFFFFFFF, tl.uint32),
        tl.full([BLOCK_N], 0x80000000, tl.uint32),
    )
    mono = score_bits ^ xor_mask
    mono = tl.where(in_bounds, mono, tl.zeros([BLOCK_N], tl.uint32))

    threshold = tl.zeros([BLOCK_N], tl.uint32)
    for i in tl.static_range(0, 32):
        candidate = threshold | tl.full([BLOCK_N], 1 << (31 - i), tl.uint32)
        count = tl.sum((mono >= candidate).to(tl.int32))
        accept = count >= topk
        threshold = tl.where(accept, candidate, threshold)

    strict_mask = mono > threshold
    tie_mask = mono == threshold

    strict_count = tl.sum(strict_mask.to(tl.int32))
    remaining = topk - strict_count

    packed = (strict_mask.to(tl.uint32) << 16) | tie_mask.to(tl.uint32)
    packed_prefix = tl.cumsum(packed)
    strict_prefix = (packed_prefix >> 16).to(tl.int32)
    tie_prefix = (packed_prefix & 0xFFFF).to(tl.int32)

    valid_tie = tie_mask & (tie_prefix <= remaining)
    final_mask = strict_mask | valid_tie

    write_pos = tl.where(strict_mask, strict_prefix - 1, strict_count + tie_prefix - 1)

    scatter_ptrs = topk_indices_ptr + pid_b * stride_out_b + write_pos.to(tl.int64) * stride_out_k
    tl.store(scatter_ptrs, token_idx, mask=final_mask)


@torch.no_grad()
def kernel(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table,
           topk_indices):
    batch_size, H, D = q_index_fp8.shape
    num_pages, page_size, _, head_dim_sf = k_index_cache_fp8.shape
    head_dim = head_dim_sf - 4
    _, max_num_pages = block_table.shape
    topk = 2048

    page_bytes = page_size * head_dim_sf
    fp8_view = torch.as_strided(
        k_index_cache_fp8.view(torch.float8_e4m3fn),
        size=(num_pages, page_size, head_dim),
        stride=(page_bytes, head_dim, 1),
    )
    scale_view = torch.as_strided(
        k_index_cache_fp8.view(torch.float32),
        size=(num_pages, page_size),
        stride=(page_bytes // 4, 1),
        storage_offset=page_size * head_dim // 4,
    )

    if max_num_pages == 1:
        fast_small_kernel[(batch_size,)](
            seq_lens, block_table, topk_indices,
            block_table.stride(0),
            topk_indices.stride(0), topk_indices.stride(1),
            BLOCK_T=page_size, TOPK=topk,
        )
        return

    if max_num_pages <= 32:
        BLOCK_K_SCORELESS = 256
        scoreless_kernel[(batch_size, triton.cdiv(topk, BLOCK_K_SCORELESS))](
            seq_lens, block_table, topk_indices,
            block_table.stride(0), block_table.stride(1),
            topk_indices.stride(0), topk_indices.stride(1),
            page_size=page_size,
            topk=topk,
            BLOCK_K=BLOCK_K_SCORELESS,
        )
        return

    max_scored = max_num_pages * page_size
    scores = torch.empty(
        (batch_size, max_scored), device=q_index_fp8.device, dtype=torch.float32
    )

    grid = (batch_size, max_num_pages)
    score_kernel[grid](
        q_index_fp8, fp8_view, scale_view, weights, seq_lens, block_table, scores,
        q_index_fp8.stride(0), q_index_fp8.stride(1), q_index_fp8.stride(2),
        fp8_view.stride(0), fp8_view.stride(1), fp8_view.stride(2),
        scale_view.stride(0), scale_view.stride(1),
        weights.stride(0), weights.stride(1),
        block_table.stride(0), block_table.stride(1),
        scores.stride(0), scores.stride(1),
        BLOCK_H=H, BLOCK_D=D, BLOCK_T=page_size,
    )

    BLOCK_N = triton.next_power_of_2(max_scored)
    radix_topk_kernel[(batch_size,)](
        scores, seq_lens, block_table, topk_indices,
        scores.stride(0), scores.stride(1),
        block_table.stride(0), block_table.stride(1),
        topk_indices.stride(0), topk_indices.stride(1),
        max_scored, max_num_pages,
        page_size=page_size,
        topk=topk,
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )
