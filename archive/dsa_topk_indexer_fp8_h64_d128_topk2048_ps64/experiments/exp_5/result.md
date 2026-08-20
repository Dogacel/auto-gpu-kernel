# Experiment 5 — 2026-04-16

**Description:** Set `num_warps=8` on the score kernel launch (up from
Triton's default of 4).

## Results
- Pass: (quick) 2/2 exact match
- A/B vs exp 4 (paired same-VM, stride 8):
  - B wins 11/16 (marginal)
  - mean Δ = −0.0006 ms (−0.17%)
  - per-workload range: −0.59% to +0.39%
- Verdict: **marginal / essentially neutral**. Reverted.

## Learnings
- num_warps doesn't matter for this kernel. Work per program is
  tiny: one 64×128 × 128×64 fp8 MMA + a few loads. Neither 4 nor 8
  warps hide enough latency to matter, and there's no loop to
  pipeline.
- Launch-overhead-bound regime: per-program work << launch latency,
  so no amount of micro-tuning inside the kernel will move the needle
  much. Gains must come from (a) reducing the grid, (b) reducing the
  number of torch post-ops, or (c) fusing post-ops into the kernel.
- Profile before more micro-tunes.
