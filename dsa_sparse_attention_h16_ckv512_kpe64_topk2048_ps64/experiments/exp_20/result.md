# Experiment 20 — 2026-04-17

**Description:** Two-part experiment per plan.md. Action 1 (primary): enable Blackwell CTA clustering on `_fused_split_combine_kernel` via `num_ctas=NUM_SPLITS` to co-locate the 8 polling CTAs per token on a single GPC, compressing the atomic-barrier spin-wait floor. Action 2 (fallback): reorder `idx_scan` load before Q loads in `_fused_attn_kernel` (T≤2 path) to overlap HBM latency.

## Results

### Action 1 — Cluster launch: **COMPILE FAILURE**

Two sub-attempts:

1. **Plan's literal recipe** (`num_ctas=NUM_SPLITS` + `cluster_dims=(1, NUM_SPLITS, 1)`) → `KeyError: cluster_dims unrecognised`. Probing Modal's Triton 3.6.0 `CUDAOptions.__dataclass_fields__` confirmed: `cluster_dims` is NOT an option on this Triton build. Only `num_ctas` exists. The driver hardcodes `clusterDim.x = num_ctas, .y = 1, .z = 1` (`triton/backends/nvidia/driver.py:352-354`).

2. **Axis-swapped to X-cluster** (`grid=(1, num_tokens)` + `num_ctas=NUM_SPLITS`, swap `s = program_id(0)` / `t = program_id(1)`):
   ```
   PlanCTA.cpp:212: Assertion `!tiled && "CTA tiling is already determined"' failed.
   note: Pipeline failed while executing [`TritonGPUPlanCTAPass` on 'builtin.module' operation]
   RuntimeError: PassManager::run failed
   ```
   Triton's CTA planner can't tile the tensor ops for clustered execution in this kernel. The atomic
   ops (release/acquire `atomic_add` on counter; volatile `tl.load`; atomic decrement) and/or the
   mixed Q-load + K-load + softmax tile shapes create ambiguity that the planner rejects.

Both sub-attempts ABORTED.

### Action 2 — Q-load overlap: **NEUTRAL** (reverted)

Two A/B runs vs exp_18: 8/12 B wins, mean Δ ≈ 0 ms. All deltas <1.6%, consistent with noise.

Critical observation: the 8/12 wins are concentrated on **large-T workloads where the change doesn't apply** (`_fused_split_combine_kernel` was untouched). The small-T workloads `0c23b10c` (T=1, valid=2) and `b7668cfd` (T=1) consistently regress +0.18 to +0.41% — the exact workloads where reorder should help. Q-load reorder is at best neutral on small-T.

Conclusion: compiler's default scheduling already overlaps Q and indices loads optimally. Triton's `num_stages=2` prefetch is sufficient; manually issuing `idx_scan` first changes the scheduler's op order but doesn't reduce the critical path.

Reverted.

## Decision: **REVERTED. Kernel returns to exp_18 state.**

Both actions failed:
- Action 1 hit a Triton 3.6 compiler limitation unrelated to the algorithm (CTA planner rejects the op mix).
- Action 2 found no measurable benefit on the intended path.

## Learnings

- **Triton 3.6's CTA planner rejects kernels with atomic cross-CTA synchronization.** The
  `TritonGPUPlanCTAPass` (`PlanCTA.cpp:212`) asserts `!tiled` — i.e., each tensor op must be assigned
  a single CTA tiling decision, but our atomic-barrier + Q/K/V tile + softmax mix produces
  conflicting tiling requirements that the planner can't resolve. Empirically this means cluster
  launch via `num_ctas` is not currently viable for any kernel containing `atomic_add` on a
  cross-CTA counter. A future experiment could try stripping the atomic barrier out of a
  cluster-launched variant and replacing it with cluster-barrier PTX via `tl.inline_asm_elementwise`
  — but this is a single coupled change (new launch layout + new barrier impl + per-kernel CTA
  planner nudges via layouts/constraints) and is much riskier than the exp_20 plan anticipated.

- **Triton 3.6 on Modal does not expose `cluster_dims`.** Only `num_ctas` is accepted; the
  driver implicitly chooses a 1D X-axis cluster of `num_ctas` size. Plans that reference
  `cluster_dims` or `cluster_dims=(1,NUM_SPLITS,1)` need to be rewritten for axis-swap.

- **Compiler's default schedule is already good for small-T Q-load overlap.** Reordering
  `idx_scan` before Q loads in `_fused_attn_kernel` produced no measurable saving (noise-level
  Δ on both A/B runs). With `num_stages=2`, Triton already pipelines independent loads. Manual
  reorder at the Python level does not strictly change the LLVM/PTX scheduling.

## Next directions

Cluster-sync is the only remaining large-T lever per profile.md but is now confirmed blocked by
Triton 3.6's CTAPlanner for this kernel shape. Remaining paths:

1. **Warp specialization** within a single CTA (persistent warps as consumers, producer warps
   for K/V loads). This keeps each CTA's internal pipeline but could cut barrier cost if the
   producer-consumer overlap saves >0.5 µs. High complexity in Triton.

2. **Attempt cluster-sync with a fresh kernel** that avoids atomic_add entirely. Replace the
   barrier with grid-wide cooperative sync via the `launch_cooperative_grid` option we just
   discovered in CUDAOptions. This needs NUM_SPLITS*num_tokens ≤ sm_count for deadlock safety
   (64 ≤ 148 on B200). Reset of the counter is unnecessary with cooperative sync.

3. **Pivot to Gluon.** Given we've exhausted Triton 3.6's scalar + structural knobs
   without landing a >1% win in 5 experiments (exp_15 through exp_19), a Gluon rewrite is the
   recommended next direction per `CLAUDE.md`. Gluon gives lower-level primitives (explicit TMA,
   warp-specialization, cluster barriers) that Triton 3.6's frontend doesn't expose. Will consider
   after 1-2 more Triton experiments.
