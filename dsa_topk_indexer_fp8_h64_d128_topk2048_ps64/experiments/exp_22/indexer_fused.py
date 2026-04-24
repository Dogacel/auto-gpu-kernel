"""
DSA TopK Indexer — exp 22 snapshot (REVERTED): two-dot mp=2 fast path.

Produced +1.7 µs mean regression (mp=2 workloads at 69 µs vs 25 µs default).
Kept for reference; see exp_22/result.md.

Omitted for brevity: fast_small_kernel, score_kernel, remap_kernel (identical
to exp 20). The mp2 variant is the change under test:
"""
import torch
import triton
import triton.language as tl


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
    BLOCK_T_PER_PAGE: tl.constexpr,  # 64
    BLOCK_T_TOTAL: tl.constexpr,     # 128
    TOPK: tl.constexpr,
):
    # REVERTED: 69 µs per mp=2 workload (vs 25 µs default path).
    # Tested the hypothesis that exp 21's regression was due to fp8 SHMEM
    # shuffle; turns out two-dot + fp32 combine costs the same. The overhead
    # is not in fp8 layout conversion — likely in tl.sort at BLOCK_N=128 or
    # in register pressure from two simultaneous K tiles.
    pid_b = tl.program_id(0)

    seq_len = tl.load(seq_lens_ptr + pid_b)
    page_0 = tl.load(block_table_ptr + pid_b * stride_btb + 0 * stride_btp).to(tl.int64)
    page_1 = tl.load(block_table_ptr + pid_b * stride_btb + 1 * stride_btp).to(tl.int64)

    h_offs = tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, BLOCK_D)
    t_offs_pp = tl.arange(0, BLOCK_T_PER_PAGE)

    q_off = pid_b * stride_qb + h_offs[:, None] * stride_qh + d_offs[None, :] * stride_qd
    q_fp8 = tl.load(q_ptr + q_off)

    w = tl.load(w_ptr + pid_b * stride_wb + h_offs * stride_wh)

    # Page 0: independent [64, 128] @ [128, 64] MMA → [64, 64] scores.
    k0_off = page_0 * stride_kp + t_offs_pp[:, None] * stride_kt + d_offs[None, :] * stride_kd
    k0 = tl.load(k_fp8_ptr + k0_off)
    s0 = tl.load(k_scale_ptr + page_0 * stride_ksp + t_offs_pp * stride_kst)
    scores_0 = tl.dot(q_fp8, tl.trans(k0), out_dtype=tl.float32)
    scores_0 = tl.maximum(scores_0, 0.0) * w[:, None]
    final_0 = tl.sum(scores_0, axis=0) * s0  # [64] fp32

    # Page 1: independent [64, 128] @ [128, 64] MMA → [64, 64] scores.
    k1_off = page_1 * stride_kp + t_offs_pp[:, None] * stride_kt + d_offs[None, :] * stride_kd
    k1 = tl.load(k_fp8_ptr + k1_off)
    s1 = tl.load(k_scale_ptr + page_1 * stride_ksp + t_offs_pp * stride_kst)
    scores_1 = tl.dot(q_fp8, tl.trans(k1), out_dtype=tl.float32)
    scores_1 = tl.maximum(scores_1, 0.0) * w[:, None]
    final_1 = tl.sum(scores_1, axis=0) * s1  # [64] fp32

    # Combine two [64] fp32 vectors into concatenated [128] — 256 B of data only.
    final_joined = tl.join(final_0, final_1)   # [64, 2]
    final_perm = tl.trans(final_joined, 1, 0)  # [2, 64]
    final = tl.reshape(final_perm, [BLOCK_T_TOTAL])  # [128]

    t_offs = tl.arange(0, BLOCK_T_TOTAL)
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


# Python dispatch branch used during exp 22:
#
# if max_num_pages == 2:
#     fast_small_kernel_mp2[(batch_size,)](
#         q_index_fp8, fp8_view, scale_view, weights,
#         seq_lens, block_table, topk_indices,
#         q_index_fp8.stride(0), q_index_fp8.stride(1), q_index_fp8.stride(2),
#         fp8_view.stride(0), fp8_view.stride(1), fp8_view.stride(2),
#         scale_view.stride(0), scale_view.stride(1),
#         weights.stride(0), weights.stride(1),
#         block_table.stride(0), block_table.stride(1),
#         topk_indices.stride(0), topk_indices.stride(1),
#         BLOCK_H=H, BLOCK_D=D,
#         BLOCK_T_PER_PAGE=page_size, BLOCK_T_TOTAL=2 * page_size,
#         TOPK=topk,
#     )
#     return
