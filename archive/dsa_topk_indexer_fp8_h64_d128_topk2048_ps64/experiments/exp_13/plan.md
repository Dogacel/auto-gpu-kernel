# Experiment 13 — Dynamic effective_topk + shrunk score_kernel grid via `seq_lens.max().item()`

## Goal

Shrink two things per-workload:

1. **`torch.topk` input**: reduce K from fixed 2048 to `min(2048, max(seq_lens))`, and reduce the scanned width from `max_num_pages * 64` to `ceil(max_sl / 64) * 64`. Cost of K=2048 sort that finds mostly -1e30 padding is the single biggest phase on workloads with short sequences.
2. **`score_kernel` grid**: reduce the outer `pid_p` axis from `max_num_pages` to `ceil(max_sl / 64)`. On workloads where `max_sl << max_num_pages*64`, this cuts per-call program count by up to 3-5× (beyond what exp-9 early-return already buys).

## Workload evidence (from `experiments/workload_profile.md`)

- **69/128 workloads** have all batch rows with `sl < 2048` — so even when `effective_topk == 2048`, topk is sorting a padded row.
- **27 workloads** could use K ≤ 1024 (2× cut), **13 workloads** K ≤ 128 (16× cut), **8** K ≤ 64.
- Mean `early_return_frac = 70.9%`. The workload at the 90th percentile has 90% of programs dispatching for nothing.
- `max_num_pages` p50 = 32 but 22 workloads are in the [82..91] tail.
- **No workload has utilization above 67%** — always shrinkable.

## Approach

```python
max_scored = max_num_pages * page_size

# Only pay the .item() sync when it can actually help: when the natural cap
# max_scored already >= topk, there's nothing to shrink on torch.topk K.
if max_scored > topk:
    max_sl = int(seq_lens.max().item())  # 1 sync
    grid_p = max(1, min((max_sl + page_size - 1) // page_size, max_num_pages))
    max_scored_eff = grid_p * page_size
    effective_topk = min(topk, max(1, max_sl))
else:
    grid_p = max_num_pages
    max_scored_eff = max_scored
    effective_topk = max_scored

scores = torch.empty((batch_size, max_scored_eff), ..., dtype=torch.float32)

grid = (batch_size, grid_p)
score_kernel[grid](..., BLOCK_T=page_size)

_, topk_idx = torch.topk(scores, effective_topk, dim=-1)
# remap_kernel already accepts effective_topk as runtime arg — no kernel changes.
```

Kernel internals untouched (exp-10 kernel body preserved: early-return +
scale-after-sum). The grid shrink is a pure Python-side change that
happens to also shrink `scores` allocation and `torch.topk` work.

## Risks

- **`.item()` sync cost**: ~3-10 µs per call. On workloads where
  `max_scored <= topk`, skipped (no sync). On workloads where the sync
  is paid, savings on torch.topk + smaller grid must exceed it.
  - Rough math: sync 5 µs; torch.topk savings ~5-15 µs on small-K
    workloads; grid savings 2-5 µs where `grid_p < max_num_pages`. Net
    expected +0 to +10 µs on most, +15-20 µs on the smallest workloads.
- **scores shape mismatch**: the kernel writes to `scores[b, pid_p*64 + ...]`
  with `pid_p < grid_p`. Bounded — no OOB store.
- **topk correctness**: scores is sliced to `max_scored_eff`. All active
  scores (below any batch's seq_len) fit in `[0, max_scored_eff)` since
  no batch has `seq_len > max_sl`. Padding positions (`seq_len <= pos < max_sl`)
  hold `-1e30` from exp-9 early-return / in-bounds mask. torch.topk
  skips them. `effective_topk` caps output at max_sl so no OOB read.

## Success criterion

- Full mean Δ ≤ −5% vs exp 10 (0.047 ms → 0.044 ms).
- A/B vs exp 10: at least 10/16 wins, mean Δ ≤ −3%.
- Correctness: 128/128 exact match.
