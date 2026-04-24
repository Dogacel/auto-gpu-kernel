---
exp: 22
date: 2026-04-17
status: reverted
parent: exp_20
---

# Result — Fast path for `max_num_pages == 2` via two independent dots + fp32 combine (reverted, regression)

## Change

Added `fast_small_kernel_mp2` per `exp_22/plan.md`: two independent
`[64, 128] @ [128, 64]` MMAs, each reduced to its own `[64]` fp32
vector before combining via small `tl.join + tl.trans + tl.reshape`
on fp32 `[64]` data (256 B total). Hypothesis per plan.md: avoiding
the 16 KB fp8 shuffle from exp 21 would fix the regression.

## Results
- Pass: **128/128** exact match
- Kernel latency (ms): min=0.0080 / mean=0.0513 / median=0.0530 / max=0.0780
- **Regression: +1.7 µs mean vs exp 20 (0.0496 ms), essentially identical to exp 21's 0.0513.**
- Mode: full + quick

## Per-workload breakdown on mp=2 targets

Same 5 workloads that regressed in exp 21 regressed **identically** here:

| uuid | exp 20 | exp 21 (big-MMA) | exp 22 (two-dot) |
|---|---:|---:|---:|
| cd594d26 | 0.025 | 0.070 | **0.069** |
| b2098949 | 0.024 | 0.069 | **0.069** |
| 83cb81c5 | 0.026 | 0.069 | **0.068** |
| d54c1568 | 0.025 | 0.069 | **0.068** |
| 13dad24c | 0.026 | 0.069 | **0.067** |

**The two different implementations produce indistinguishable 70 µs results.**

## Why the hypothesis was wrong

The plan assumed the fp8 tile shuffle (exp 21's `tl.join` on 16 KB
fp8) was the bottleneck. Exp 22 eliminated that shuffle entirely —
yet ran at the same latency. The ~45 µs overhead (vs default path's
~25 µs) is therefore NOT in the fp8 layout conversion.

Candidate root causes ruled in/out:
- **Fp8 SHMEM shuffle**: Ruled out. Exp 22 doesn't do one.
- **Two K loads (2 × 16 KB from HBM)**: Both exps pay this. Possible.
- **Small grid (5 programs on 148 SMs)**: Both exps share this. Possible — single-wave launch leaves ~145 SMs idle, but launch overhead should dominate at ~5-10 µs not 45 µs.
- **BLOCK_N=128 `tl.sort`**: Both exps use this. Previous lesson said "OK up to ~512" but `probably`. Possible.
- **Register pressure from 2-MMA live state + sort staging buffer**: Both exps pay this.
- **Kernel JIT compile on first trial**: Unlikely — baseline also JITs; benchmark averages over 5 trials.

Exp 22 is a clean negative control for the fp8-shuffle hypothesis. The overhead must live in one of the commonalities above — most likely the `tl.sort` at BLOCK_N=128 and/or register pressure from live two-page state.

## Lesson

**Two-dot fp32-combine is NOT the fix for mp=2 regressions.** Both
exp 21 (big-MMA with fp8 shuffle) and exp 22 (two-dot with fp32
combine) land at ~70 µs per program, vs ~25 µs for the default 3-kernel
dispatch path. The shared factor is not the fp8 layout but something
else in the fused-small-kernel pattern. Specifically ruled out: fp8
tile SHMEM rearrangement is **not the primary cost**.

Next candidate investigation: profile or ablation to isolate the
cost of `tl.sort` at BLOCK_N=128 vs BLOCK_N=64. If sort@128 costs
~45 µs (compared to ~1-2 µs at BLOCK_N=64 from exp 20), then
BLOCK_N=128 is effectively past a bitonic-sort wall that the earlier
lesson didn't detect.

## Reverted to exp 20

Next steps:
- Launch `research` agent — plateau trigger (2 consecutive same-axis reverts, exp 21+22; no new direction obvious without data).
- The agent should look at the workload profile for mp=2 workloads, compare exp 20 vs the two regressed fast paths, and propose either a sort-ablation experiment or a different mp=2 strategy (e.g., keep the default 3-kernel path for mp=2, but accelerate the remap side, or find a way to reduce the 25→15 µs baseline without the fused-kernel trap).
