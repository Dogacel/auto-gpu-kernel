---
exp: 20
date: 2026-04-17
status: new_best
parent: exp_10
---

# Result — Workload-specialized fast path for `max_num_pages == 1`

## Change

Added `fast_small_kernel` Triton kernel and host-side branch in `kernel()`:

```python
if max_num_pages == 1:
    fast_small_kernel[(batch_size,)](...)  # fused score+sort+remap in one launch
    return
# else: exp 10 path (score_kernel + torch.topk + remap_kernel)
```

`max_num_pages = block_table.shape[1]` is a pure Python value, no GPU sync.

The fast kernel, for each batch `b` (grid `(B,)` — one program per batch):
1. Loads `seq_len[b]`, `block_table[b, 0] → page_id`, Q [64, 128], K [64, 128], scale [64], w [64].
2. Computes scores [64] via exp-10 formula: `sum_h(relu(q·k.T) * w) * scale`.
3. Masks positions ≥ seq_len to −1e30.
4. Sorts via packed uint64 (`(mono_f32 << 32) | idx`) on BLOCK_T=64 (well below tl.sort's 2048 wall).
5. Writes `topk_indices`: first a full `[2048]` store of −1, then overwrite first 64 with `tl.where(i < actual_topk, page_id*64 + sorted_idx, −1)`.

## Results
- Pass: **128/128** exact match
- Kernel latency (ms) on this VM run: min=0.0080 / mean=0.0496 / median=0.0525 / max=0.0780
- Reference latency: n/a (ref reports 0.000 ms on all — correctness assessed via `matched_ratio=1.0000` across 5 trials)
- Max abs err: 0.00 | Max rel err: 0.00 (exact match)
- Mode: full 128-workload + ab-vs-exp_10 (stride-8, n=16)

## A/B vs exp 10 (paired, same VM)

```
Paired n=16 | B wins 5/16 | mean Δ = -0.0009 ms → B faster
```

| uuid | A (ms) | B (ms) | ΔB−A (ms) | % |
|---|---:|---:|---:|---:|
| **30cecff1** | 0.0247 | 0.0094 | **-0.0153** | **-61.91%** |
| bb22d09a | 0.0529 | 0.0529 | -0.0000 | -0.08% |
| df80c00b | 0.0487 | 0.0487 | -0.0000 | -0.01% |
| e515e20a | 0.0527 | 0.0527 | -0.0000 | -0.03% |
| 05775386 | 0.0488 | 0.0488 | -0.0000 | -0.01% |
| (all others: ±0.0002 ms within noise) |

Only 30cecff1 in the stride-8 sample hits the fast path (15 of 128 workloads have max_pg=1 per workload_profile.md; stride-8 sampling caught one). Non-fast-path workloads are tied within ±0.0002 ms — the Python branch `if max_num_pages == 1` costs effectively nothing on the default path.

On the full run, **8 workloads land at 0.008-0.009 ms** (the fast path slot): 30cecff1, 4667f9ad, 46f236c0, 9410ad1e, e64a4ebc, 752c2ee5, d0c00dd5, abc9d12c. Most of these were at ~0.025-0.030 ms under exp 10.

## Why it won

Per `experiments/profile.md` on 30cecff1: py_setup 12 µs + alloc 9 µs + score_kernel 26 µs + torch.topk 26 µs + remap_kernel 19 µs = 93 µs end-to-end. These workloads have 64 real tokens max, so torch.topk and remap_kernel are pure dispatch overhead for < 5 µs of actual top-K work.

The fused kernel collapses score+sort+remap into a single launch:
- Skips `torch.topk` dispatch (≈25 µs).
- Skips `remap_kernel` launch (≈19 µs).
- Also skips the `scores` tensor allocation (pre-allocated `topk_indices` is the only buffer).

Net: ~15 µs per fast-path workload (observed 24.7 µs → 9.4 µs on 30cecff1). Kernel-side compute is small (one MMA + one 64-wide sort + one 2048-int store).

## Correctness notes

- Tie-breaking: torch.topk and packed-descending sort may differ on ties, but real scores come from fp32 dot products (ties extremely rare) and masked/padded positions (−1e30) all fall outside `actual_topk = min(seq_len, 64)` so their tie-break order is invisible in the output. Verified: 128/128 exact match.
- `seq_len ≤ 64` is guaranteed when `max_num_pages == 1` (pages are 64-token), so `actual_topk ≤ 64` and all real indices fit in the first-64 overwrite region.

## Learnings

1. **Host-known shape branching is nearly free** and can unlock regime-specific fused paths. `max_num_pages = block_table.shape[1]` is a Python int from `torch.Tensor.shape` — no GPU round-trip, no `.item()` sync. This is fundamentally different from exp 13/14 (which tried to branch on `.item()`-synced values and lost 30-60 µs per sync).
2. **Fused score+sort+remap is viable at small N.** `tl.sort` on BLOCK_N=64 is trivial; the bitonic wall is at 2048+. This is the regime where Triton sort beats `torch.topk` because dispatch is the cost, not compute.
3. **Full [2048] −1 store then overwrite [64]** is cheaper than masking per-element — the hardware handles sequential stores fine, and skips branch logic inside the kernel.
4. **8 fast-path hits, not 15, in this run.** Either the plan miscounted or some `max_pg=1` workloads are hashed differently. Still a real win, and it generalizes — extending the fast path to `max_num_pages ≤ 2` could capture more workloads (see LESSONS).

## Next

- **Fast path for `max_num_pages == 2`**: covers a larger slice of small workloads (estimated ~20-30 more per workload_profile.md). Needs a small-N variant of score_kernel with BLOCK_T=128 (same layout question as exp 18 but without the medium-workload trap, since these are all short-seq workloads).
- Profile the `max_pg=1` fast path under `profiler` agent to find the remaining 9 µs (py_setup presumably dominates now).
