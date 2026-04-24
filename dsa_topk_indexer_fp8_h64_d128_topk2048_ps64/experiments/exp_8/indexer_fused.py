"""
DSA TopK Indexer — Triton score + topk + remap kernels (exp 8, REVERTED).

Added a topk_kernel that uses tl.sort on a packed (monotone_f32_bits, index) uint64 key.
Branched: use triton top-K for BLOCK_N ≤ 2048, fall back to torch.topk for larger.
Result: net neutral / slight regression (mean Δ ≈ +0 in A/B, +3.8% in full).
Reverted in favor of exp 7 kernel.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def score_kernel(
    q_ptr, k_fp8_ptr, k_scale_ptr, w_ptr,
    seq_lens_ptr, block_table_ptr, scores_ptr,
    stride_qb, stride_qh, stride_qd,
    stride_kp, stride_kt, stride_kd,
    stride_ksp, stride_kst,
    stride_wb, stride_wh,
    stride_btb, stride_btp,
    stride_sb, stride_st,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_p = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + pid_b)
    token_start = pid_p * BLOCK_T
    tile_active = token_start < seq_len

    page_id_raw = tl.load(block_table_ptr + pid_b * stride_btb + pid_p * stride_btp)
    page_id = tl.where(tile_active, page_id_raw, 0).to(tl.int64)

    h_offs = tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, BLOCK_D)
    t_offs = tl.arange(0, BLOCK_T)

    q_off = pid_b * stride_qb + h_offs[:, None] * stride_qh + d_offs[None, :] * stride_qd
    q_fp8 = tl.load(q_ptr + q_off)

    k_off = page_id * stride_kp + t_offs[:, None] * stride_kt + d_offs[None, :] * stride_kd
    k_fp8 = tl.load(k_fp8_ptr + k_off)

    scores = tl.dot(q_fp8, tl.trans(k_fp8), out_dtype=tl.float32)

    s_off = page_id * stride_ksp + t_offs * stride_kst
    scale = tl.load(k_scale_ptr + s_off)
    scores = scores * scale[None, :]

    scores = tl.maximum(scores, 0.0)

    w_off = pid_b * stride_wb + h_offs * stride_wh
    w = tl.load(w_ptr + w_off)
    scores = scores * w[:, None]

    final = tl.sum(scores, axis=0)

    abs_t = token_start + t_offs
    in_bounds = abs_t < seq_len
    final = tl.where(in_bounds, final, -1e30)

    score_off = pid_b * stride_sb + abs_t * stride_st
    tl.store(scores_ptr + score_off, final)


@triton.jit
def topk_kernel(
    scores_ptr,          # float32 [B, max_scored]
    topk_idx_ptr,        # int64 [B, effective_topk]
    stride_sb, stride_st,
    stride_ib, stride_ik,
    max_scored,
    effective_topk,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)

    offs = tl.arange(0, BLOCK_N)
    mask = offs < max_scored

    score_ptrs = scores_ptr + pid_b * stride_sb + offs * stride_st
    scores = tl.load(score_ptrs, mask=mask, other=-float('inf'))

    # Float32 → monotone uint32: for f≥0: bits ^ 0x80000000 (flip sign bit);
    # for f<0: bits ^ 0xFFFFFFFF (flip all bits). Preserves ordering.
    score_bits = scores.to(tl.uint32, bitcast=True)
    sign = score_bits >> 31
    mask_low = tl.zeros_like(score_bits) - sign  # 0xFFFFFFFF if sign=1, 0 if sign=0
    xor_mask = mask_low | tl.full([BLOCK_N], 0x80000000, tl.uint32)
    mono = score_bits ^ xor_mask
    # Invert index in low bits so descending sort breaks ties by ascending index.
    inv_idx = (BLOCK_N - 1 - offs).to(tl.uint32)
    packed = (mono.to(tl.uint64) << 32) | inv_idx.to(tl.uint64)

    sorted_packed = tl.sort(packed, dim=0, descending=True)
    sorted_idx = (BLOCK_N - 1) - (sorted_packed & 0xFFFFFFFF).to(tl.int32)

    out_mask = offs < effective_topk
    out_ptrs = topk_idx_ptr + pid_b * stride_ib + offs * stride_ik
    tl.store(out_ptrs, sorted_idx.to(tl.int64), mask=out_mask)


@triton.jit
def remap_kernel(
    topk_idx_ptr,        # int64 [B, topk]
    block_table_ptr,     # int32 [B, max_num_pages]
    seq_lens_ptr,        # int32 [B]
    topk_indices_ptr,    # int32 [B, topk]  (DPS output)
    stride_idx_b, stride_idx_k,
    stride_bt_b, stride_bt_p,
    stride_out_b, stride_out_k,
    page_size: tl.constexpr,
    max_num_pages,
    effective_topk,
    topk: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)

    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_in_topk = k_offs < topk
    k_in_effective = k_offs < effective_topk

    seq_len = tl.load(seq_lens_ptr + pid_b)
    actual_topk = tl.minimum(seq_len.to(tl.int32), effective_topk)

    idx_ptrs = topk_idx_ptr + pid_b * stride_idx_b + k_offs * stride_idx_k
    topk_idx = tl.load(idx_ptrs, mask=k_in_effective, other=0)

    page_idx = topk_idx // page_size
    offset = topk_idx % page_size

    page_idx_clamped = tl.minimum(page_idx, max_num_pages - 1)

    bt_ptrs = block_table_ptr + pid_b * stride_bt_b + page_idx_clamped * stride_bt_p
    global_page = tl.load(bt_ptrs, mask=k_in_effective, other=0).to(tl.int64)

    token_idx = (global_page * page_size + offset).to(tl.int32)

    in_range = k_offs < actual_topk
    final = tl.where(in_range, token_idx, -1)

    out_ptrs = topk_indices_ptr + pid_b * stride_out_b + k_offs * stride_out_k
    tl.store(out_ptrs, final, mask=k_in_topk)


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

    effective_topk = min(topk, max_scored)

    BLOCK_N = triton.next_power_of_2(max_scored)
    if BLOCK_N <= 2048:
        topk_idx = torch.empty(
            (batch_size, effective_topk), device=q_index_fp8.device, dtype=torch.int64
        )
        topk_kernel[(batch_size,)](
            scores, topk_idx,
            scores.stride(0), scores.stride(1),
            topk_idx.stride(0), topk_idx.stride(1),
            max_scored, effective_topk,
            BLOCK_N=BLOCK_N,
        )
    else:
        _, topk_idx = torch.topk(scores, effective_topk, dim=-1)

    BLOCK_K = 256
    remap_grid = (batch_size, triton.cdiv(topk, BLOCK_K))
    remap_kernel[remap_grid](
        topk_idx, block_table, seq_lens, topk_indices,
        topk_idx.stride(0), topk_idx.stride(1),
        block_table.stride(0), block_table.stride(1),
        topk_indices.stride(0), topk_indices.stride(1),
        page_size=page_size,
        max_num_pages=max_num_pages,
        effective_topk=effective_topk,
        topk=topk,
        BLOCK_K=BLOCK_K,
    )
