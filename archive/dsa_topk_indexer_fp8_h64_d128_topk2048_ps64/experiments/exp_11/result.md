---
exp: 11
date: 2026-04-17
status: reverted
parent: exp_10
---

# Result — num_warps=8 retry (post exp-9/10 kernel changes)

## Change
Added `num_warps=8` to the `score_kernel[grid](...)` launch.
Unchanged otherwise.

## Measurement

A/B vs exp 10 (paired, same VM):

```
Paired n=16 | B wins 7/16 | mean Δ = +0.0001 ms → A faster
```

Most workloads essentially tied (±0.05%). Two outliers:
- `30cecff1` (smallest, B=1 likely): **+12.79%** slower with
  num_warps=8.
- `e515e20a`: −2.28% faster.

Net: no aggregate movement, and a clear regression on the smallest
workload class. Reverted.

## Why num_warps=8 doesn't help here
- Score_kernel body post-exp-9 is already small on inactive
  programs (early-return stores a constant and exits). Those
  programs don't need more warps.
- Active programs do a 64×128 FP8 dot + 64-element sum. Triton's
  default num_warps (4) is already tuned for this tile size. More
  warps means fewer threads per output element → smaller register
  footprint per thread but more scheduling pressure.
- Very small workloads (B=1) launch ~max_num_pages programs
  total. With num_warps=8 each uses 2x the warp budget, which
  likely reduces occupancy on small grids.

## Reverted to exp 10 state

Kernel file restored to exp 10 (scale-after-sum, no num_warps
override). Proceed to exp 12 on a different axis.

## Takeaway
Per LESSONS.md: num_warps=8 was already known to be a wash on
the pre-fused kernel (exp 5). Post-fusion it's actively harmful
on small workloads. Default (num_warps=4) is the right call for
the 64×128 FP8 tile shape. Don't revisit unless the tile changes.
