# Experiment 4 — 2026-04-16

**Description:** Tried NUM_SPLITS=16 (double exp_2's 8) to increase SM utilization on T=1/T=2 workloads. Combine kernel re-tuned: switched from `tl.static_range` to dynamic `range(NUM_SPLITS)` with `num_stages=2` pipelining (first attempt with `tl.static_range(16)` exploded to 0.125 ms due to heavy register pressure from 16× unrolled loads of 32 KB acc_si tiles).

Reverted exp_3's `if`-based early-exit (it broke pipelining; see exp_3/result.md).

## Results
- Pass: 12/12
- Kernel latency (ms): small=0.028 / large=0.036 / overall=0.028 (min) / 0.036 (median) / 0.036 (max)
- Reference latency (ms): not profiled
- Max abs err: 1.56e-02
- Mode: stride 2 (12 workloads)
- vs exp_2 (NUM_SPLITS=8, best): **regression across the board** — small +33% (0.021→0.028), large +50% (0.024→0.036).

**Per-workload deltas:**
| UUID | exp_2 | exp_4 | Δ |
|---|---|---|---|
| 0c23b10c (T=1) | 0.021 | 0.028 | +33% ❌ |
| b7668cfd (T=2) | 0.022 | 0.031 | +41% ❌ |
| 05f6de65 | 0.023 | 0.032 | +39% ❌ |
| 4c46a94b | 0.024 | 0.036 | +50% ❌ |
| 02d6ae9c..2207f0fd (T=6-8) | 0.024 | 0.036 | +50% ❌ |

## Learnings
- **NUM_SPLITS=16 is worse than 8 at every workload size.** Even T=1 (where more splits should help) regresses: the combine-kernel cost grows linearly with NUM_SPLITS (more loads of partial_m/l/acc + more online-softmax merge iters), and the split-kernel itself gets less work per CTA (2048/16=128 vs 2048/8=256 TopK elements, only 2 BLOCK_N=64 blocks vs 4 — not enough to amortize prologue overhead).
- **`tl.static_range(16)` on a body that loads 32 KB (16×512 f32 acc_si) per iter is a mistake.** The 16-way unroll inflates kernel bytecode + register pressure → 5× regression on top of the NUM_SPLITS doubling. Use dynamic `range(...)` for loops where the body is large; Triton's `num_stages=2` prefetcher handles the pipelining.
- **The combine kernel is a real cost.** With NUM_SPLITS=8, `combine` is cheap (8 small sequential online-softmax merges). At 16, it roughly doubles, which eats the split-K parallelism win. Sweet spot is ≤ 8 for this problem size; going higher needs a structurally different combine (tree reduction? parallel reduce along NUM_SPLITS via grid-y?).

**Next move (exp_5):** Revert `NUM_SPLITS=16→8`, but KEEP the dynamic combine loop with `num_stages=2` (vs exp_2's default). This isolates whether the combine change alone has any effect at the original NUM_SPLITS=8 baseline.

## Sub-lesson for LESSONS.md
`tl.static_range(N)` with large N and heavy per-iter work (e.g., loading multi-KB tiles) causes register/icache pressure regressions — Triton inlines everything. Prefer dynamic `range()` with `num_stages>1` for such loops.
