"""
DSA TopK Indexer — Triton score kernel + Triton remap kernel (exp 17).
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
    SCALE_OFFSET: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_p = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + pid_b)
    token_start = pid_p * BLOCK_T

    if token_start >= seq_len:
        t_offs_sk = tl.arange(0, BLOCK_T)
        score_off_sk = pid_b * stride_sb + (token_start + t_offs_sk) * stride_st
        tl.store(scores_ptr + score_off_sk, tl.full([BLOCK_T], -1e30, tl.float32))
        return

    page_id_raw = tl.load(block_table_ptr + pid_b * stride_btb + pid_p * stride_btp)
    page_id = page_id_raw.to(tl.int64)

    h_offs = tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, BLOCK_D)
    t_offs = tl.arange(0, BLOCK_T)

    q_off = pid_b * stride_qb + h_offs[:, None] * stride_qh + d_offs[None, :] * stride_qd
    q_fp8 = tl.load(q_ptr + q_off)

    k_off = page_id * stride_kp + t_offs[:, None] * stride_kt + d_offs[None, :] * stride_kd
    k_fp8 = tl.load(k_fp8_ptr + k_off)

    # Scale offset baked in as constexpr — previously applied via storage_offset in as_strided.
    s_off = page_id * stride_ksp + SCALE_OFFSET + t_offs * stride_kst
    scale = tl.load(k_scale_ptr + s_off)
    w_off = pid_b * stride_wb + h_offs * stride_wh
    w = tl.load(w_ptr + w_off)

    scores = tl.dot(q_fp8, tl.trans(k_fp8), out_dtype=tl.float32)

    scores = tl.maximum(scores, 0.0)
    scores = scores * w[:, None]

    final = tl.sum(scores, axis=0) * scale

    abs_t = token_start + t_offs
    in_bounds = abs_t < seq_len
    final = tl.where(in_bounds, final, -1e30)

    score_off = pid_b * stride_sb + abs_t * stride_st
    tl.store(scores_ptr + score_off, final)


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
    batch_size = q_index_fp8.shape[0]
    max_num_pages = block_table.shape[1]

    H: int = 64
    D: int = 128
    page_size: int = 64
    topk: int = 2048
    page_bytes: int = 8448
    scale_stride_p: int = 2112
    scale_offset: int = 2048

    k_fp8 = k_index_cache_fp8.view(torch.float8_e4m3fn)
    k_scale = k_index_cache_fp8.view(torch.float32)

    max_scored = max_num_pages * page_size
    scores = torch.empty(
        (batch_size, max_scored), device=q_index_fp8.device, dtype=torch.float32
    )

    grid = (batch_size, max_num_pages)
    score_kernel[grid](
        q_index_fp8, k_fp8, k_scale, weights, seq_lens, block_table, scores,
        q_index_fp8.stride(0), q_index_fp8.stride(1), q_index_fp8.stride(2),
        page_bytes, D, 1,
        scale_stride_p, 1,
        weights.stride(0), weights.stride(1),
        block_table.stride(0), block_table.stride(1),
        scores.stride(0), scores.stride(1),
        SCALE_OFFSET=scale_offset,
        BLOCK_H=H, BLOCK_D=D, BLOCK_T=page_size,
    )

    effective_topk = min(topk, max_scored)
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
