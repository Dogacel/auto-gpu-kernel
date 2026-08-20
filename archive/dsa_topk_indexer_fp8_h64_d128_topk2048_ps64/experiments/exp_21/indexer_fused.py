"""
DSA TopK Indexer — Triton score kernel + Triton remap kernel (exp 10),
with workload-specialized fast paths for max_num_pages == 1 (exp 20)
and max_num_pages == 2 (exp 21 — ATTEMPTED, reverted: regression).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fast_small_kernel(
    q_ptr, k_fp8_ptr, k_scale_ptr, w_ptr,
    seq_lens_ptr, block_table_ptr, topk_indices_ptr,
    stride_qb, stride_qh, stride_qd,
    stride_kp, stride_kt, stride_kd,
    stride_ksp, stride_kst,
    stride_wb, stride_wh,
    stride_btb,
    stride_out_b, stride_out_k,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    TOPK: tl.constexpr,
):
    # Fused score + sort + remap for workloads with max_num_pages == 1.
    pid_b = tl.program_id(0)
    seq_len = tl.load(seq_lens_ptr + pid_b)
    page_id = tl.load(block_table_ptr + pid_b * stride_btb).to(tl.int64)

    h_offs = tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, BLOCK_D)
    t_offs = tl.arange(0, BLOCK_T)

    q_off = pid_b * stride_qb + h_offs[:, None] * stride_qh + d_offs[None, :] * stride_qd
    q_fp8 = tl.load(q_ptr + q_off)

    k_off = page_id * stride_kp + t_offs[:, None] * stride_kt + d_offs[None, :] * stride_kd
    k_fp8 = tl.load(k_fp8_ptr + k_off)

    scale = tl.load(k_scale_ptr + page_id * stride_ksp + t_offs * stride_kst)
    w = tl.load(w_ptr + pid_b * stride_wb + h_offs * stride_wh)

    scores = tl.dot(q_fp8, tl.trans(k_fp8), out_dtype=tl.float32)
    scores = tl.maximum(scores, 0.0)
    scores = scores * w[:, None]
    final = tl.sum(scores, axis=0) * scale

    in_bounds = t_offs < seq_len
    final = tl.where(in_bounds, final, -1e30)

    bits = final.to(tl.uint32, bitcast=True)
    xor_mask = ((bits >> 31) * 0x7FFFFFFF) | 0x80000000
    mono = bits ^ xor_mask
    packed = (mono.to(tl.uint64) << 32) | t_offs.to(tl.uint64)
    sorted_packed = tl.sort(packed, descending=True)
    sorted_idx = (sorted_packed & 0xFFFFFFFF).to(tl.int64)

    k_offs = tl.arange(0, TOPK)
    out_ptrs = topk_indices_ptr + pid_b * stride_out_b + k_offs * stride_out_k
    tl.store(out_ptrs, tl.full([TOPK], -1, tl.int32))

    actual_topk = tl.minimum(seq_len.to(tl.int32), BLOCK_T)
    t_offs_out = tl.arange(0, BLOCK_T)
    token_idx = (page_id * BLOCK_T + sorted_idx).to(tl.int32)
    final_idx = tl.where(t_offs_out < actual_topk, token_idx, -1)
    real_out_ptrs = topk_indices_ptr + pid_b * stride_out_b + t_offs_out * stride_out_k
    tl.store(real_out_ptrs, final_idx)


@triton.jit
def fast_small_kernel_mp2(
    q_ptr, k_fp8_ptr, k_scale_ptr, w_ptr,
    seq_lens_ptr, block_table_ptr, topk_indices_ptr,
    stride_qb, stride_qh, stride_qd,
    stride_kp, stride_kt, stride_kd,
    stride_ksp, stride_kst,
    stride_wb, stride_wh,
    stride_btb, stride_btp,
    stride_out_b, stride_out_k,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T_PER_PAGE: tl.constexpr,
    BLOCK_T_TOTAL: tl.constexpr,
    TOPK: tl.constexpr,
):
    # REGRESSED: this kernel costs ~70 µs vs ~25 µs for the default path.
    # The tl.join + tl.trans + tl.reshape on 16 KB fp8 data is not amortized
    # on a small grid (1 program per batch). Kept for reference; see exp 22.
    pid_b = tl.program_id(0)

    seq_len = tl.load(seq_lens_ptr + pid_b)
    page_0 = tl.load(block_table_ptr + pid_b * stride_btb + 0 * stride_btp).to(tl.int64)
    page_1 = tl.load(block_table_ptr + pid_b * stride_btb + 1 * stride_btp).to(tl.int64)

    h_offs = tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, BLOCK_D)
    t_offs_pp = tl.arange(0, BLOCK_T_PER_PAGE)
    t_offs = tl.arange(0, BLOCK_T_TOTAL)

    q_off = pid_b * stride_qb + h_offs[:, None] * stride_qh + d_offs[None, :] * stride_qd
    q_fp8 = tl.load(q_ptr + q_off)

    k0_off = page_0 * stride_kp + t_offs_pp[:, None] * stride_kt + d_offs[None, :] * stride_kd
    k1_off = page_1 * stride_kp + t_offs_pp[:, None] * stride_kt + d_offs[None, :] * stride_kd
    k0 = tl.load(k_fp8_ptr + k0_off)
    k1 = tl.load(k_fp8_ptr + k1_off)

    k_joined = tl.join(k0, k1)
    k_perm = tl.trans(k_joined, 2, 0, 1)
    k_stacked = tl.reshape(k_perm, [BLOCK_T_TOTAL, BLOCK_D])

    s0 = tl.load(k_scale_ptr + page_0 * stride_ksp + t_offs_pp * stride_kst)
    s1 = tl.load(k_scale_ptr + page_1 * stride_ksp + t_offs_pp * stride_kst)
    s_joined = tl.join(s0, s1)
    s_perm = tl.trans(s_joined, 1, 0)
    scale = tl.reshape(s_perm, [BLOCK_T_TOTAL])

    w = tl.load(w_ptr + pid_b * stride_wb + h_offs * stride_wh)

    scores = tl.dot(q_fp8, tl.trans(k_stacked), out_dtype=tl.float32)
    scores = tl.maximum(scores, 0.0)
    scores = scores * w[:, None]
    final = tl.sum(scores, axis=0) * scale

    in_bounds = t_offs < seq_len
    final = tl.where(in_bounds, final, -1e30)

    bits = final.to(tl.uint32, bitcast=True)
    xor_mask = ((bits >> 31) * 0x7FFFFFFF) | 0x80000000
    mono = bits ^ xor_mask
    packed = (mono.to(tl.uint64) << 32) | t_offs.to(tl.uint64)
    sorted_packed = tl.sort(packed, descending=True)
    sorted_idx = (sorted_packed & 0xFFFFFFFF).to(tl.int64)

    k_offs = tl.arange(0, TOPK)
    out_ptrs = topk_indices_ptr + pid_b * stride_out_b + k_offs * stride_out_k
    tl.store(out_ptrs, tl.full([TOPK], -1, tl.int32))

    within_page = sorted_idx & (BLOCK_T_PER_PAGE - 1)
    page_sel = tl.where(sorted_idx < BLOCK_T_PER_PAGE, page_0, page_1)
    token_idx = (page_sel * BLOCK_T_PER_PAGE + within_page).to(tl.int32)

    actual_topk = tl.minimum(seq_len.to(tl.int32), BLOCK_T_TOTAL)
    t_offs_out = tl.arange(0, BLOCK_T_TOTAL)
    final_idx = tl.where(t_offs_out < actual_topk, token_idx, -1)
    real_out_ptrs = topk_indices_ptr + pid_b * stride_out_b + t_offs_out * stride_out_k
    tl.store(real_out_ptrs, final_idx)


# score_kernel and remap_kernel: identical to exp 10 — omitted from this
# snapshot for brevity. See solution/triton/indexer_fused.py or exp_20/.
