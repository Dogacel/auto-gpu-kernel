---
exp: 21
date: 2026-04-17
status: reverted
parent: exp_20
---

# Result — Fast path for `max_num_pages == 2` via concatenated [128, 128] MMA (reverted, regression)

## Change

Added `fast_small_kernel_mp2` kernel and branch `if max_num_pages == 2`
in the host dispatch. The kernel mirrors exp 20's design but for two
pages:

- Load two K pages (`k0, k1 [64, 128]` each), concat via `tl.join +
  tl.trans + tl.reshape` → `[128, 128]` fp8 tile.
- One `[64, 128] @ [128, 128]` MMA → `[64, 128]` scores.
- Combine per-page scales via the same join/trans/reshape pattern on
  `[64]` fp32 → `[128]` scale vector.
- Standard relu + w_mul + sum + mask + packed-uint64 sort (BLOCK_N=128).
- Remap: `page_sel = sorted_idx < 64 ? page_0 : page_1`,
  `within_page = sorted_idx & 63`.

## Results
- Pass: **128/128** exact match
- Kernel latency (ms) on full run: min=0.0080 / mean=0.0512 / median=0.0525 / max=0.0780
- **Regression: +1.6 µs mean vs exp 20.**
- Mode: full + quick

## Per-workload breakdown

mp=2 workloads **regressed sharply** (the exact set we were targeting):

| uuid | exp 20 (default path) | exp 21 (new fast path) | Δ |
|---|---:|---:|---:|
| cd594d26 | 0.025 ms | **0.070 ms** | **+0.045 ms (+180%)** |
| b2098949 | 0.024 ms | **0.069 ms** | **+0.045 ms** |
| 83cb81c5 | 0.026 ms | **0.069 ms** | **+0.043 ms** |
| d54c1568 | 0.025 ms | **0.069 ms** | **+0.044 ms** |
| 13dad24c | 0.026 ms | **0.069 ms** | **+0.043 ms** |

mp=1 workloads (exp 20 baseline, untouched): still at 0.008 ms.

Net impact: 5 workloads × +44 µs = +220 µs total, ÷ 128 = **+1.7 µs mean**.

## Why it lost

The `tl.join → tl.trans(2, 0, 1) → tl.reshape([128, 128])` rearrangement
is on **16 KB of fp8 data** (two `[64, 128]` tiles). On a large grid
(exp 18: `B * max_num_pages` programs, ~2000+), this cost amortizes
over launch setup and gets merged into the prologue. On our **5-to-15
program grid** (one program per batch, B ≤ 15 for mp=2 workloads),
there's no amortization — the SHMEM shuffle to materialize the stacked
fp8 tile dominates.

Theory agrees: exp 18 saw +5-7% on medium workloads from this same
pattern (tl.trans of `[64, 128, 2]` fp8). In that case the regression
was masked by the grid-halving benefit. Here we *only* have the
rearrangement cost, no grid-halving benefit (grid goes from `2*B` to
`B`, not from `2*B` to fewer than B). So the absolute cost is 45 µs
per program, which is roughly the total work the default exp-10 path
does on these workloads in the first place.

## Lesson

**tl.join + tl.trans + tl.reshape on 16 KB fp8 tiles is expensive
(~30-45 µs per program in isolation)** — not a free compile-time
rearrangement. It scales poorly on small grids because the setup
cost is not amortized.

For concatenating K tiles across pages:
- **Big-MMA concat pattern is only viable when grid is large AND
  provides grid-halving or tile-size benefits.** For 1-program-per-batch
  fast paths, pick a different pattern.
- **Alternative: two independent `[64, 64]` dots, accumulate to separate
  `[64]` fp32 vectors, then combine the fp32 scalars (256 B, trivial).**
  This avoids the 16 KB fp8 shuffle entirely. Exp 22 plan.

## Reverted to exp 20

Next candidate: **exp 22 = mp=2 fast path via two-dot pattern** (load
both pages, do two [64, 64] MMAs, combine the two `[64]` fp32 vectors
via a tiny join+reshape). This isolates the compute from the fp8
layout conversion cost.
