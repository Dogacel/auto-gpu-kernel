# Experiment 52 — 2026-04-17

**Description:** Change combine loop from `tl.static_range(NUM_SPLITS)` to `range(NUM_SPLITS)` on line 148 of `_fused_split_combine_kernel`. Motivated by profile.md recommendation ("2-stage combine tree" + LESSON-13 threshold at 32 KB unroll). Test if dynamic range with auto-pipelining beats static unrolling at NUM_SPLITS=16.

Baseline: exp_51 (NUM_SPLITS=16 new best).

Hypothesis: At NUM_SPLITS=16 with BLOCK_D=32 (2 KB per partial_acc tile), 16 iters × 2 KB = 32 KB of unrolled code — at LESSON-13's reported threshold. Dynamic range should reduce icache pressure and enable auto-pipelining.

## Results
- Pass: 2/2 (quick mode; A/B was enough to disqualify)
- Kernel latency: +5-7% regression on all 7 T≥3 workloads; T≤2 noise
- Mode: A/B stride 2 × 1 run (single regression run sufficient)

### Stratified A/B (single run, A=exp_51 B=exp_52 range variant)

| Class | Workloads | Δ% | Mechanism |
|---|---|---|---|
| T≥3 (7/8 workloads) | 02d6ae9c, 2207f0fd, 232ed014, 4c46a94b, 5096e459, 564007ac, 78b2e11c | +5-7% | Lost ILP in dynamic combine loop |
| T=8 max-valid (1) | 05f6de65 | -0.28% | Tied |
| T≤2 (4) | 0c23b10c, b7668cfd, e6b849f2, f77df5ce | ±1% | Unchanged path |

Mean Δ = +0.0005 ms → A (static_range) faster.

### Mechanism — why dynamic range regressed

Dynamic `range(16)` trades compile-time unrolling for runtime pipelining. At 2 KB/iter, per-iter work is small:
- 1 load: `partial_acc[si, :, offs_d]` = H×BLOCK_D fp32 = 2 KB
- 1 softmax rescale: ~30 FP32 ops  
- 1 vector multiply-add: `acc_comb * alpha + acc_si * beta` = 512 fused ops

Static unrolling lets the compiler interleave 16 copies of the above — loads from iter 2 overlap with compute of iter 1, etc. Full ILP exploits the 8 warps × 32 threads.

Dynamic range doesn't know the iter count at compile time, so it serializes the loop body sequentially. Triton's auto-pipeliner (via num_stages) provides SOME overlap but can only prefetch 1-2 iters ahead vs static unroll's 16-wide ILP.

The regression (~0.8 µs = 15% of combine time) confirms combine is compute-latency-bound for the rescale chain, not IO-bound. Static unroll reduces critical path by parallelizing across iters.

## Learnings

**LESSON-13 refinement**: 32 KB unroll is the **upper bound** for static_range efficacy, not the crossover point. Below 32 KB (and possibly including it), static_range remains preferred even at 16 iters. The exp_4 original failure (0.125 ms) was at **32 KB tile × 16 iters = 512 KB** total static code, far beyond this case.

**Combine-loop unroll axis well-characterized now:**
- NUM_SPLITS=8 × static_range: exp_48 baseline (5.18 µs)
- NUM_SPLITS=16 × static_range: exp_51 current (5.12 µs — essentially same despite 2× iters)
- NUM_SPLITS=16 × range: exp_52 this test (+0.8 µs, worse)
- NUM_SPLITS=8 × range: not tested; expected +0.5 µs based on trend

**Key insight**: combine's bottleneck is NOT unroll footprint but softmax-rescale critical path length. To accelerate combine, need to parallelize the rescale (tree reduction or warp-level split of splits), not just change the loop sugar.

## Decision
Revert to exp_51 (static_range). Combine-loop axis largely closed — remaining lever is structural tree reduction (per profile.md recommendation), which is a bigger kernel rewrite.
