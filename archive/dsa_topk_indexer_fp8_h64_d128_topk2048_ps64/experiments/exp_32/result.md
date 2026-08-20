---
exp: 32
date: 2026-04-17
status: reverted
parent: exp_29
---

# Experiment 32 — 2026-04-17

**Description:** Retry `num_warps=8` on `score_kernel` post structural changes
(exp 9 early-return, exp 10 scale-after-sum, exp 20/25/26 fast paths).
Exp 11's regression was mostly on `30cecff1` (now on `fast_small_kernel`, not
score_kernel), so the old verdict may no longer apply.

## Implementation

Single-line change: `num_warps=8` on `score_kernel[grid](...)`.

## Results

- Pass: (quick skipped, A/B covers both)
- A/B vs exp 29 (stride 8, paired, same VM):
  - B wins 7/16, mean Δ = **+0.0000 ms** → A faster (tied)
  - Slow-path: a876010b -0.5% (improved marginally), f457feb2 +2.7% (regressed)
  - No clear pattern; mostly noise
- Mode: ab-vs-exp_29
- **Reverted.**

## Learnings

1. **Confirms LESSONS.md rule of thumb.** `num_warps=8` helps reduction-heavy
   kernels (radix_topk's 32× tl.sum on BLOCK_N=8192). score_kernel does a single
   MMA + a tl.sum over 64 heads — no BLOCK_N≥2048 reduction, so doubling warps
   adds scheduling overhead without parallelization gain.
2. **Post-exp 29 doesn't change the verdict for score_kernel.** The kernel body
   is still structurally: load Q, load K, MMA, relu, multiply weights, sum_h,
   scale, write. No reduction loop, no multi-MMA chain. Exp 11's refuted
   verdict is still valid.
3. **Profile's "WEAK / NEEDS RE-TEST" marker was a false positive cue.** The
   30cecff1 regression in exp 11 was workload-specific (mp=1, single tile),
   which is now on fast_small_kernel. The other 15 workloads also regressed.
   The exp 11 A/B was more uniform than remembered.

## Next candidates

- **Fusion** remains the biggest ceiling per profile, but risks MMA parallelism
  loss. Need a design that preserves (B, mp) grid parallelism while eliminating
  the HBM scores round-trip. Bit-histogram-based radix done cross-grid via
  atomics could preserve MMA parallelism but has high complexity.
- **Merge two `tl.cumsum` calls** in radix into one via priority encoding —
  unclear if profitable without specific scheduling gain.
- Explore **Gluon** — the profile's fusion proposal (register-backed per-batch
  scores buffer with streaming MMAs per tile) fits Gluon's programming model
  better than Triton's. Not yet triggered per /optimize skill (needs 15-20
  iterations without improvement; we're at 3 since exp 29).
