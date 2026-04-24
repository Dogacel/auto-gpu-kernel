"""
DSA TopK Indexer — Triton score kernel + batched PyTorch top-K (exp 4).
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


@torch.no_grad()
def kernel(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table,
           topk_indices):
    batch_size, H, D = q_index_fp8.shape
    num_pages, page_size, _, head_dim_sf = k_index_cache_fp8.shape
    head_dim = head_dim_sf - 4
    _, max_num_pages = block_table.shape
    topk = 2048

    # SOA: [P, page_size*head_dim_sf] uint8 = [P, 8192 fp8 bytes | 256 scale bytes]
    kv_u8 = k_index_cache_fp8.view(torch.uint8).reshape(num_pages, page_size * head_dim_sf)
    fp8_view = (
        kv_u8[:, :page_size * head_dim]
        .contiguous()
        .view(num_pages, page_size, head_dim)
        .view(torch.float8_e4m3fn)
    )
    scale_view = (
        kv_u8[:, page_size * head_dim:]
        .contiguous()
        .view(num_pages, page_size, 4)
        .view(torch.float32)
        .squeeze(-1)
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

    device = q_index_fp8.device
    effective_topk = min(topk, max_scored)
    _, topk_idx = torch.topk(scores, effective_topk, dim=-1)

    page_idx_per_token = (topk_idx // page_size).clamp_(max=max_num_pages - 1)
    offset_per_token = topk_idx % page_size

    bt_long = block_table.to(torch.long)
    global_page_idx = torch.gather(bt_long, 1, page_idx_per_token)
    topk_tokens = (global_page_idx * page_size + offset_per_token).to(torch.int32)

    seq_lens_long = seq_lens.to(torch.long)
    actual_topks = torch.minimum(
        seq_lens_long, torch.full_like(seq_lens_long, effective_topk)
    )
    arange = torch.arange(effective_topk, device=device).unsqueeze(0)
    mask = arange < actual_topks.unsqueeze(-1)
    masked = torch.where(mask, topk_tokens, torch.full_like(topk_tokens, -1))

    topk_indices.fill_(-1)
    topk_indices[:, :effective_topk].copy_(masked)
