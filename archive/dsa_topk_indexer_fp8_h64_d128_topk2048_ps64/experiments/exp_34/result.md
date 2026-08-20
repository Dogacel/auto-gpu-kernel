---
exp: 34
date: 2026-04-17
status: reverted
parent: exp_33
---

# Experiment 34 — 2026-04-17

**Description:** Try `num_warps=16` (up from 8) on `radix_topk_kernel` now
that exp 33's cumsum merge freed a tree reduction — hypothesis was that lower
register pressure might make higher warp count viable.

## Implementation

Single-line change: `num_warps=8` → `num_warps=16` on `radix_topk_kernel` launch.

## Results

- Pass: 2/2 quick
- A/B vs exp 33 (stride 8, paired same-VM):
  - B wins 4/16, mean Δ = **+0.0001 ms** → A faster
  - **Medium slow-path workloads regressed +3-5%** (19e7663d +3.5%, 2f3b7321
    +4.6%, de54c4e6 +4.8%, e63194e7 +4.7%, 4c7705ad +2.0%)
  - Large (a876010b) +0.3% — within noise
  - f457feb2 -11.44% and f457feb2 in exp 33 baseline was 0.0225 vs now 0.0199
    — the reference shift flags VM-level variance on that one workload
- Mode: ab-vs-exp_33
- **Reverted.**

## Learnings

1. **num_warps=8 is the sweet spot for this kernel.** 16 warps doubles
   inter-warp reduction stages (+1 tree level) and likely doubles scheduler
   contention on B200's 4 schedulers/SM (16 warps / 4 = 4 per scheduler vs
   8/4=2 for num_warps=8). Net: +3-5% regression on medium BLOCK_N=4096
   reductions.
2. **Register pressure wasn't the limiter.** The hypothesis that exp 33's
   cumsum merge freed registers and unlocked higher warp count is refuted —
   the reduction tree depth is more important than occupancy at num_warps≥8.
3. **Axis is saturated.** num_warps=4 was worse (exp 28 → 29 improvement),
   num_warps=8 is best, num_warps=16 is worse. Ternary search has found its
   optimum; further tuning here is wasted.

## Next candidates

- Masked block_table load in radix (mask=final_mask instead of mask=in_bounds)
  — reduces block_table HBM traffic from BLOCK_N=8192 loads to ~2048 per
  batch. Minor (~0.1-0.3 µs full-run mean) but clean test.
- Fusion still the biggest ceiling (~8 µs per slow workload) — needs a design
  that works within Triton's constraints (no register slice-assign on tiles).
- Score_kernel HBM bandwidth gap: 66 GB/s effective vs 103 GB/s memcpy floor.
  Potential 5 µs/slow workload win if HBM access patterns can be improved —
  diagnose via profiler before guessing.
