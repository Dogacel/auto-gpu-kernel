# Experiment 19 — 2026-04-17

**Description:** Try `num_stages=3` on the fused_split_combine_kernel (keep num_warps=8). Hypothesis: deeper pipelining might hide some K-load latency if split loop has 2 iters. Low-risk experiment.

## Results
- Pass: 12/12 A/B (correctness ok)
- Mode: A/B vs exp_18 (same-VM paired)
- **A/B: B (exp_19) wins 3/12, mean Δ = +0.0005 ms** → clear regression
- Large T: +5.78 to +6.44% across all 7 workloads (+0.0009 to +0.0010 ms)

## Decision: **Reverted.**

## Why it regressed

Split loop has 0-2 iters (SPLIT_SIZE=256, BLOCK_N=128). `num_stages=3` allocates 3× shmem
for K tiles (3 × 128KB = 384KB), exceeding B200's 228KB-per-SM shmem budget. This
forces either:
- Register spills to local memory (slower),
- Reduced occupancy (more CTAs/SM capped),
- Fallback to smaller effective K-tile (wasted shmem).

For 0-2 iter loops, `num_stages>2` has no compute benefit (nothing to pipeline beyond
iter 1). So any shmem cost is pure downside.

## Learnings

- **num_stages > iter_count brings pure shmem cost, no compute benefit.** For our
  `for bn in range(0, max_bn, BLOCK_N)` split loop where max_bn ≤ 2×BLOCK_N, cap
  num_stages at 2. Higher values trigger B200 shmem overflows that degrade occupancy
  or cause register spills.
- **Both extremes of num_stages (1 not tested, but 3 regresses) and num_warps ({4,16})
  now regress.** The structure is tightly tuned — num_warps=8, num_stages=2, BLOCK_N=128,
  NUM_SPLITS=8. Further wins require structural changes.

## Next directions

Scalar tuning is clearly exhausted. The remaining structural levers:
1. **Cluster-sync via `num_ctas=NUM_SPLITS` + inline-PTX `barrier.cluster.*`** — projected −0.5 to −1 µs on large T after exp_18's volatile-load already cut 0.2 µs of poll cost.
2. **Small-T specialization**: e.g., skip the scan of all 2048 indices via progressive scan, Q-load + index-load overlap.
3. **Reduce Q-load redundancy**: 8 CTAs per token each load the same Q; cluster DSMEM could share. Requires cluster launch.
