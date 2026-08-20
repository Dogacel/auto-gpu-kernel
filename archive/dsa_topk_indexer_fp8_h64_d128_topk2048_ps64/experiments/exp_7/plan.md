# Experiment 7 — Fuse remap+mask into a single Triton kernel

## Goal

Replace the 10+ torch post-ops with a single `remap_kernel` launch.
Target saving ~40–60 µs of per-iteration launch overhead and op
dispatch cost.

## Current post-kernel path (after exp 6)

```python
_, topk_idx = torch.topk(scores, effective_topk, dim=-1)          # kept
page_idx_per_token = (topk_idx // page_size).clamp_(max=...)      # kernel 1
offset_per_token = topk_idx % page_size                           # kernel 2
bt_long = block_table.to(torch.long)                              # kernel 3
global_page_idx = torch.gather(bt_long, 1, page_idx_per_token)    # kernel 4
topk_tokens = (global_page_idx * page_size + offset_per_token)    # kernel 5
          .to(torch.int32)                                        # kernel 6
seq_lens_long = seq_lens.to(torch.long)                           # kernel 7
actual_topks = torch.minimum(seq_lens_long, full_like(...))       # kernels 8–9
arange = torch.arange(effective_topk, device=device).unsqueeze(0) # kernel 10
mask = arange < actual_topks.unsqueeze(-1)                        # kernel 11
masked = torch.where(mask, topk_tokens, full_like(-1))            # kernels 12–13
topk_indices.fill_(-1)                                            # kernel 14
topk_indices[:, :effective_topk].copy_(masked)                    # kernel 15
```

~15 torch launches, each ~5 µs of CPU→GPU dispatch latency on B200.
Profile put the remap+mask phases at ~75 µs total; this includes
compute, but the bulk of that is launch overhead for tiny ops.

## New path

```python
_, topk_idx = torch.topk(scores, effective_topk, dim=-1)   # kept
remap_kernel[grid](                                        # single kernel replaces all below
    topk_idx, block_table, seq_lens, topk_indices,
    page_size, max_num_pages, topk, ...
)
```

Kernel per-program work:
- Load 1 tile of `topk_idx[b, k:k+BLOCK_K]` as int64.
- Compute `page_idx = topk_idx // page_size`, `off = topk_idx % page_size`.
- Clamp `page_idx` to `[0, max_num_pages - 1]`.
- Load `global_page = block_table[b, page_idx]` (scalar gather).
- Compute `token_idx = global_page * page_size + off`.
- Load `seq_len = seq_lens[b]`; mask positions where `k_idx >= min(topk, seq_len)` to `-1`.
- Store int32 to `topk_indices[b, k:k+BLOCK_K]`.

Grid: `(B, ceil(topk / BLOCK_K))`. With `BLOCK_K = 256`, 8 programs per
batch. For B=32 (large workloads), 256 programs — fits ~2 waves on
B200's 150 SMs. For B=1 (small), 8 programs — undersubscribed but
kernel is tiny.

If `effective_topk < topk`, positions `k >= effective_topk` must still
be `-1` in the output (DPS requirement). Handle by making the kernel
aware of `effective_topk` and writing `-1` where `k >= actual_topk_b`
for each batch (where `actual_topk_b = min(effective_topk, seq_len[b])`).

## Risks

- **Gather into block_table is serialized**: each thread in a warp
  does a unique offset load. Triton should handle via
  `tl.load(block_table_ptr + ...)` with per-thread offset.
  Block table is `[B, max_num_pages]` int32, max ~2000 pages per row
  → 8 KB per row, well within L1.
- **int64 division/modulo**: Triton supports these on indices. Cost
  should be bounded.
- **Correctness vs exp 6**: must match exactly. torch.topk ordering
  and our page remap algebra are unchanged; we just move them into
  a kernel.

## Success criterion

- A/B vs exp 6 on stride 8: B wins ≥13/16, mean Δ ≤ −15%.
- Target: mean 0.200 → 0.150 ms (~25% speedup).
- Correctness: exact match (matched_ratio = 1.0 on all 16 quick workloads).
