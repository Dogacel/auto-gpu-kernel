"""
DSA TopK Indexer — Triton score kernel + Triton topk_kernel + remap kernel (exp 15).

Replaces torch.topk (~45-52 µs, 45% of event total) with a tile-merge selection
kernel that maintains a running top-2048 in uint64-packed form and merges each
BLOCK_CHUNK=2048 chunk of scores via a 4096-element sort per chunk.
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

    # Early-return for tiles fully beyond seq_len — 94% of programs on large
    # workloads (grid is sized for max_num_pages, but most batches use far less).
    # Must still write -1e30 so torch.topk skips these positions.
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

    # Issue the scalar loads early so they overlap with matmul latency.
    s_off = page_id * stride_ksp + t_offs * stride_kst
    scale = tl.load(k_scale_ptr + s_off)
    w_off = pid_b * stride_wb + h_offs * stride_wh
    w = tl.load(w_ptr + w_off)

    scores = tl.dot(q_fp8, tl.trans(k_fp8), out_dtype=tl.float32)

    # Scale >= 0 (deep_gemm amax/fp8_max), so max(x*s,0) = s*max(x,0) and sum is
    # linear — apply scale as a scalar-per-t multiply after the cross-head sum
    # instead of a 64x64 broadcast before relu.
    scores = tl.maximum(scores, 0.0)
    scores = scores * w[:, None]

    final = tl.sum(scores, axis=0) * scale

    abs_t = token_start + t_offs
    in_bounds = abs_t < seq_len
    final = tl.where(in_bounds, final, -1e30)

    score_off = pid_b * stride_sb + abs_t * stride_st
    tl.store(scores_ptr + score_off, final)


@triton.jit
def topk_kernel(
    scores_ptr,          # float32 [B, max_scored_padded]
    topk_idx_ptr,        # int64 [B, TOPK]
    stride_sb, stride_st,
    stride_ib, stride_ik,
    max_scored,
    TOPK: tl.constexpr,         # 2048
    BLOCK_CHUNK: tl.constexpr,  # 2048
    NUM_CHUNKS: tl.constexpr,
):
    pid_b = tl.program_id(0)

    # Initial running key: all zeros. In the packed monotone form, 0 corresponds
    # to the smallest possible score (f32 -inf after monotonization sign-flip
    # gives the largest uint32 value for negative, and conversely for positive;
    # 0 maps to neither — it's below all real scores and sorts to the bottom
    # in a descending sort, which is what we want for initial padding).
    running_key = tl.zeros([TOPK], tl.uint64)

    for chunk_id in tl.static_range(NUM_CHUNKS):
        base = chunk_id * BLOCK_CHUNK
        offs = base + tl.arange(0, BLOCK_CHUNK)
        mask = offs < max_scored
        s = tl.load(
            scores_ptr + pid_b * stride_sb + offs * stride_st,
            mask=mask, other=-float('inf'),
        )

        # Float32 → monotone uint32: bits ^ (sign ? 0xFFFFFFFF : 0x80000000).
        # Preserves ordering as uint comparison of bits.
        s_bits = s.to(tl.uint32, bitcast=True)
        sign = s_bits >> 31
        xor_mask = (tl.zeros_like(s_bits) - sign) | tl.full([BLOCK_CHUNK], 0x80000000, tl.uint32)
        mono = s_bits ^ xor_mask

        # Tie-break: descending sort by score, ascending by index for ties.
        # Use max_scored as inversion constant; indices in [0, max_scored).
        inv_idx = (max_scored - 1 - offs).to(tl.uint32)
        new_key = (mono.to(tl.uint64) << 32) | inv_idx.to(tl.uint64)

        # Combine running + new into a [2*TOPK] tensor: running in first half,
        # new in second half. Construct via join [TOPK] + [TOPK] → [TOPK, 2],
        # transpose to [2, TOPK], flatten. Row-major flatten of [2, TOPK] puts
        # row 0 (running) in first TOPK slots, row 1 (new) in last TOPK slots.
        joined = tl.join(running_key, new_key)                # [TOPK, 2]
        joined_T = tl.trans(joined, 1, 0)                     # [2, TOPK]
        combined = tl.reshape(joined_T, [2 * TOPK])           # [2*TOPK]
        sorted_combined = tl.sort(combined, dim=0, descending=True)
        # Take top TOPK: reshape [2*TOPK] → [2, TOPK] (row-major → row 0 is
        # first TOPK of sorted), transpose to [TOPK, 2], split → (first, second).
        sorted_2d = tl.reshape(sorted_combined, [2, TOPK])    # [2, TOPK]
        sorted_T = tl.trans(sorted_2d, 1, 0)                  # [TOPK, 2]
        top_half, _ = tl.split(sorted_T)
        running_key = top_half

    # Unpack indices from final running_key.
    sorted_idx = (max_scored - 1) - (running_key & 0xFFFFFFFF).to(tl.int32)
    out_offs = tl.arange(0, TOPK)
    tl.store(
        topk_idx_ptr + pid_b * stride_ib + out_offs * stride_ik,
        sorted_idx.to(tl.int64),
    )


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

    # SOA layout per page (bytes): [fp8_data 0..8191 | scales 8192..8447]. The [P,64,1,132]
    # shape is a reshape-only wrapper; fp8 and scales are NOT interleaved per token.
    # Build strided views directly on the underlying storage with as_strided — zero copy.
    page_bytes = page_size * head_dim_sf  # 8448
    fp8_view = torch.as_strided(
        k_index_cache_fp8.view(torch.float8_e4m3fn),
        size=(num_pages, page_size, head_dim),
        stride=(page_bytes, head_dim, 1),  # fp8 units; intra-page still contiguous
    )
    scale_view = torch.as_strided(
        k_index_cache_fp8.view(torch.float32),
        size=(num_pages, page_size),
        stride=(page_bytes // 4, 1),
        storage_offset=page_size * head_dim // 4,  # skip the 8192-B fp8 region
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

    BLOCK_CHUNK = 2048
    NUM_CHUNKS = triton.cdiv(max_scored, BLOCK_CHUNK)
    topk_idx = torch.empty(
        (batch_size, topk), device=q_index_fp8.device, dtype=torch.int64
    )
    topk_kernel[(batch_size,)](
        scores, topk_idx,
        scores.stride(0), scores.stride(1),
        topk_idx.stride(0), topk_idx.stride(1),
        max_scored,
        TOPK=topk,
        BLOCK_CHUNK=BLOCK_CHUNK,
        NUM_CHUNKS=NUM_CHUNKS,
    )

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
