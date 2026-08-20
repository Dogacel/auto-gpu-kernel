# Experiment 17 — 2026-04-17

**Description:** Try `num_warps=16` on the fused split+combine kernel (large-T path). Hypothesis from LESSON exp_14 + profile: num_warps=4 regressed 3-5%, so we've established num_warps=8 ≫ 4. Does 16 warps (1 warp per head for H=16) fare better?

## Results
- Pass: 2/2 (quick)
- Mode: quick (correctness check)
- Max abs err: 1.56e-02 (unchanged)
- **T=1 (0c23b10c): 0.005 ms** (unchanged — fused_attn_kernel path, not touched)
- **T=8 (2207f0fd): 0.018 ms** vs exp_15 baseline 0.016 ms → **+0.002 ms (+12.5%) REGRESSION**

## Decision: **Reverted.**

## Why it regressed

H=16 heads, num_warps=16 → 1 head per warp. I hypothesized this would improve softmax
reductions by eliminating cross-head coordination within warps. But measured latency
went the other way. Likely reasons:

1. **MMA tile efficiency.** Tensor-core MMA on B200 uses [16, 16, 16] or [16, 8, 16] bf16 tiles. For output shape [H=16, N=128], num_warps=8 gives each warp [2, 128] = 16 tiles of [2, 8] MMA tiles, or [16, 16] with 8-way N partition. num_warps=16 forces each warp to [1, 128] rows = 8 MMA tiles of [1, 16]... but MMA requires at least [16, 16, 16] input tile size, so the compiler may waste rows.
2. **Shared memory per CTA doubles with warp count.** Each warp needs its own scratch region for reductions, register spills, etc.
3. **Concurrency within CTA.** 16 warps × 32 threads = 512 threads = full warp saturation of a 512-thread CTA; less ILP per warp.

**Confirmed: num_warps=8 is a strict optimum for H=16 on this kernel.** Don't revisit
unless kernel structure changes (e.g., different MMA shape or head partitioning).

## Learnings

- **num_warps=8 is the strict optimum for H=16 in both directions.** num_warps=4 regresses (exp_14, cross-head serialization); num_warps=16 also regresses (this, likely MMA-tile-alignment inefficiency). 8 warps × 2 heads-per-warp is the sweet spot for this MMA shape and head count.
- **Last 3 experiments have regressed** (exp_14, exp_16, exp_17). Signal that easy tunings are exhausted at this structure; next step requires a structural change or targeted research.

## Next directions

Calling research agent for a fresh plan. Options on the table:
1. **Cluster sync via inline PTX** for Hopper+ thread-block clusters — potentially −1.5 µs on large T (biggest remaining lever per profile.md).
2. **Q-load overlap** with indices scan — potentially −0.3 to −0.8 µs on small T.
3. **Attack the 4.94 µs noop launch floor** — requires persistent kernel or CUDA-graph-like scheme (CUDA graphs forbidden per CLAUDE.md).
4. **Interleaved split indexing** — balance valid count across CTAs so they arrive at barrier simultaneously (but profile suggests most large-T workloads are already balanced).
