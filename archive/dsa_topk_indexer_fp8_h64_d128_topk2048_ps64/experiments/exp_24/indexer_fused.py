"""
DSA TopK Indexer — exp 24 snapshot (REVERTED): num_stages=3 on score_kernel.

Change: added `num_stages=3` kwarg to the `score_kernel[grid](...)` launch in
the non-fast-path dispatch. Goal: increase prefetch depth on the single-tl.dot
fp8 tile kernel so Triton overlaps the q/k/scale/w loads more aggressively.

Result: A/B vs exp 20 (paired, stride 8): B wins 9/16, mean Δ = +0.0000 ms
(tied). Per-workload deltas are all within ±1.5%, typical cross-run noise.
num_stages has no measurable effect on this kernel — consistent with the fact
that score_kernel has no outer loop (single tl.dot), so the compiler has
little to pipeline.

Reverted: `num_stages=3` removed, back to exp 20 state.
"""
import torch
import triton


@torch.no_grad()
def kernel(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table,
           topk_indices):
    # ... setup identical to exp 20 ...
    batch_size, H, D = q_index_fp8.shape
    num_pages, page_size, _, head_dim_sf = k_index_cache_fp8.shape
    head_dim = head_dim_sf - 4
    _, max_num_pages = block_table.shape
    topk = 2048

    # ... as_strided views, mp=1 fast path (unchanged from exp 20) ...

    # Exp 24 change under test (reverted):
    # score_kernel[grid](
    #     ...,
    #     BLOCK_H=H, BLOCK_D=D, BLOCK_T=page_size,
    #     num_stages=3,   # ← added, no effect vs default
    # )

    # ... rest (torch.topk + remap_kernel) identical to exp 20 ...
    return
