"""
DSA TopK Indexer — Triton score kernel + Triton remap kernel (exp 19).

Exp 19: Two pages per program via TWO independent [64×64×128] MMAs with
store-between serialization. Halves the grid vs exp 10 without the
layout-conversion cost of exp 18's concatenated-K approach. Q/w loaded
once per program.
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
    max_num_pages,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,       # per-page tile (64)
    BLOCK_T2: tl.constexpr,      # two-page span (128)
):
    pid_b = tl.program_id(0)
    pid_p = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + pid_b)
    token_start = pid_p * BLOCK_T2

    # Early-return for pairs fully beyond seq_len. Write -1e30 to all 128
    # positions so torch.topk ignores them.
    if token_start >= seq_len:
        t_offs_sk = tl.arange(0, BLOCK_T2)
        score_off_sk = pid_b * stride_sb + (token_start + t_offs_sk) * stride_st
        tl.store(scores_ptr + score_off_sk, tl.full([BLOCK_T2], -1e30, tl.float32))
        return

    # Two adjacent page indices. Clamp the second for odd max_num_pages;
    # the duplicated tail is masked to -1e30 by `abs_t < seq_len`.
    page_p0 = pid_p * 2
    page_p1 = tl.minimum(page_p0 + 1, max_num_pages - 1)

    page_id_0 = tl.load(block_table_ptr + pid_b * stride_btb + page_p0 * stride_btp).to(tl.int64)
    page_id_1 = tl.load(block_table_ptr + pid_b * stride_btb + page_p1 * stride_btp).to(tl.int64)

    h_offs = tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, BLOCK_D)
    t_offs = tl.arange(0, BLOCK_T)

    # Q and w loaded ONCE per program, shared across both page iterations.
    q_off = pid_b * stride_qb + h_offs[:, None] * stride_qh + d_offs[None, :] * stride_qd
    q_fp8 = tl.load(q_ptr + q_off)
    w_off = pid_b * stride_wb + h_offs * stride_wh
    w = tl.load(w_ptr + w_off)

    # ---------- Page 0 ----------
    k_off_0 = page_id_0 * stride_kp + t_offs[:, None] * stride_kt + d_offs[None, :] * stride_kd
    k_fp8_0 = tl.load(k_fp8_ptr + k_off_0)
    s_off_0 = page_id_0 * stride_ksp + t_offs * stride_kst
    scale_0 = tl.load(k_scale_ptr + s_off_0)

    scores_0 = tl.dot(q_fp8, tl.trans(k_fp8_0), out_dtype=tl.float32)
    scores_0 = tl.maximum(scores_0, 0.0)
    scores_0 = scores_0 * w[:, None]
    reduced_0 = tl.sum(scores_0, axis=0) * scale_0

    abs_t_0 = token_start + t_offs
    in_bounds_0 = abs_t_0 < seq_len
    final_0 = tl.where(in_bounds_0, reduced_0, -1e30)
    score_off_0 = pid_b * stride_sb + abs_t_0 * stride_st
    tl.store(scores_ptr + score_off_0, final_0)

    # ---------- Page 1 ----------
    k_off_1 = page_id_1 * stride_kp + t_offs[:, None] * stride_kt + d_offs[None, :] * stride_kd
    k_fp8_1 = tl.load(k_fp8_ptr + k_off_1)
    s_off_1 = page_id_1 * stride_ksp + t_offs * stride_kst
    scale_1 = tl.load(k_scale_ptr + s_off_1)

    scores_1 = tl.dot(q_fp8, tl.trans(k_fp8_1), out_dtype=tl.float32)
    scores_1 = tl.maximum(scores_1, 0.0)
    scores_1 = scores_1 * w[:, None]
    reduced_1 = tl.sum(scores_1, axis=0) * scale_1

    abs_t_1 = token_start + BLOCK_T + t_offs
    in_bounds_1 = abs_t_1 < seq_len
    final_1 = tl.where(in_bounds_1, reduced_1, -1e30)
    score_off_1 = pid_b * stride_sb + abs_t_1 * stride_st
    tl.store(scores_ptr + score_off_1, final_1)


@triton.jit
def remap_kernel(
    topk_idx_ptr,
    block_table_ptr,
    seq_lens_ptr,
    topk_indices_ptr,
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

    # Pad scores buffer so tail program on odd max_num_pages can store its
    # full 128-token span without OOB. Padding positions are masked to
    # -1e30 by `abs_t < seq_len` inside the kernel.
    pages_per_prog = 2
    num_p_programs = (max_num_pages + pages_per_prog - 1) // pages_per_prog
    max_scored_padded = num_p_programs * pages_per_prog * page_size
    scores = torch.empty(
        (batch_size, max_scored_padded), device=q_index_fp8.device, dtype=torch.float32
    )

    grid = (batch_size, num_p_programs)
    score_kernel[grid](
        q_index_fp8, fp8_view, scale_view, weights, seq_lens, block_table, scores,
        q_index_fp8.stride(0), q_index_fp8.stride(1), q_index_fp8.stride(2),
        fp8_view.stride(0), fp8_view.stride(1), fp8_view.stride(2),
        scale_view.stride(0), scale_view.stride(1),
        weights.stride(0), weights.stride(1),
        block_table.stride(0), block_table.stride(1),
        scores.stride(0), scores.stride(1),
        max_num_pages,
        BLOCK_H=H, BLOCK_D=D, BLOCK_T=page_size, BLOCK_T2=page_size * pages_per_prog,
    )

    effective_topk = min(topk, max_scored_padded)
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
