"""
DSA TopK Indexer — exp 23 snapshot (REVERTED): alias topk_indices as fp32 scratch.

Change: replaced `scores = torch.empty((B, max_scored), fp32)` with
`scores = topk_indices.view(torch.float32)[:, :max_scored]` when
`max_scored <= 2048` to skip the ~9 µs torch.empty dispatch per call.

Result: A/B vs exp 20: B wins 8/16, mean Δ = +0.0016 ms (A faster). 5
workloads regressed +9-13% (~+5 µs each) despite the aliasing saving
the alloc. Hypothesis: torch.topk's fast path prefers contiguous input;
the sliced strided view (stride 2048 along batch dim for
max_scored<2048) forces a slower codepath.

Omitted for brevity: fast_small_kernel, score_kernel, remap_kernel
(identical to exp 20). The dispatch branch is the change under test:
"""
import torch
import triton


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
        # ... exp 20 fast_small_kernel dispatch unchanged ...
        return

    # Exp 23 change under test:
    max_scored = max_num_pages * page_size
    if max_scored <= topk:
        # Alias DPS output buffer as fp32 scratch. Saves the 9 µs torch.empty
        # dispatch. But the [:, :max_scored] sliced view has stride (2048, 1)
        # instead of (max_scored, 1), which slows down torch.topk on
        # mp∈[2,32]\{32} workloads by ~5 µs — net regression.
        scores = topk_indices.view(torch.float32)[:, :max_scored]
    else:
        scores = torch.empty(
            (batch_size, max_scored), device=q_index_fp8.device, dtype=torch.float32
        )

    # ... score_kernel + torch.topk + remap_kernel identical to exp 20 ...
