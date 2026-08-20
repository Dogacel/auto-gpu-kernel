# Plan — exp 20

## Diagnosis
Scalar-tuning is exhausted: exp_14 (num_warps=4), exp_16 (vectorized combine), exp_17 (num_warps=16), exp_19 (num_stages=3) all regressed; only exp_18 (volatile-load poll) marginally won at −0.6% mean. Current best is exp_18 @ ~16.43 µs on large-T with the atomic-barrier fused split+combine. Per `profile.md` the only remaining *meaningful* lever on large-T is the barrier-spin bucket (2.23 µs, 14% of total); all other recoverable buckets are below 0.5 µs. The exp_18 result showed that cheapening the *per-iter poll cost* only recovers ~10% of the spin bucket — the rest is **wait-floor** (slowest CTA wins), which can only be reduced by physically co-locating the 8 polling CTAs so their shared counter line stays hot in one GPC's L1/L2.

## Strategy
**Pivot (low-risk variant of cluster-sync).** Add Blackwell thread-block clustering to the *existing* atomic-barrier kernel. Change is confined to the launch site: `grid=(num_tokens, 1)` + `cluster_dims=(1, NUM_SPLITS, 1)` + `num_ctas=NUM_SPLITS`. The kernel body is unchanged; `tl.program_id(1)` still returns `0..7` because the cluster dimension multiplies the grid. The atomic counter line now lives in the cluster's local L2 rail rather than bouncing across GPCs.

This is explicitly a *less ambitious* cluster experiment than "replace the spin with `barrier.cluster.arrive/wait` PTX": it does not change the barrier code, does not use DSMEM, does not touch correctness-sensitive surfaces. It gives us two data points in one experiment: (a) does cluster launch compile+run on this kernel, and (b) does co-locating CTAs measurably reduce spin-wait wall-clock. If (a) fails or (b) regresses, we revert a 3-line diff and fall back to the small-T Q-load overlap (Action 2 below).

**I am deliberately not recommending full cluster-sync PTX this iteration** because it couples a grid-layout change (new) with a barrier-implementation change (new PTX) in one experiment, which violates the one-change-per-iteration rule and has no safe fallback inside the same kernel. Defer full cluster-barrier to exp_21 *only if exp_20 lands*.

## Actions (priority ordered)

1. **What:** Enable 8-way CTA clustering on `_fused_split_combine_kernel` (large-T path only). Change only the launch site in `sparse_fused.py:356-378`:

   ```python
   # before
   grid = (num_tokens, NUM_SPLITS)
   _fused_split_combine_kernel[grid](
       ..., num_warps=8, num_stages=2,
   )

   # after
   grid = (num_tokens, 1)                  # cluster dim multiplies, final is (T, NUM_SPLITS)
   _fused_split_combine_kernel[grid](
       ...,
       num_warps=8, num_stages=2,
       num_ctas=NUM_SPLITS,                # must equal product(cluster_dims)
       cluster_dims=(1, NUM_SPLITS, 1),    # 8 CTAs co-scheduled per token
   )
   ```

   The kernel body (including `tl.program_id(0)` for token, `tl.program_id(1)` for split `s`) is untouched — the driver multiplies `gridDimY *= clusterDimY` before launch (`triton/backends/nvidia/driver.py:319`), so the launched grid is still `(T, NUM_SPLITS)` with per-CTA program IDs identical to today.

   **Why:** With `num_ctas=8` the 8 split CTAs for a given token are placed on the **same GPC** (CUDA_CLUSTER_SCHEDULING_POLICY_SPREAD is set by the driver, line 355; spread across clusters, packed within a cluster). The `Counter_ptr + t` line stays hot in one GPC's L2 rail; the atomic `release`/volatile-load sees ~2× lower latency. Net effect: the spin-wait floor (currently ~2 µs) compresses because the wait-observation latency drops.

   **Impact:** Expected −0.3 to −0.8 µs on large T (16.4 → 15.6–16.1). Conservative vs profile's full-cluster-barrier projection (−1.5 to −1.8 µs) because we're only recovering observation-latency, not replacing the spin with hardware-native sync. If the compiled kernel can use DSMEM intrinsics automatically (unlikely but possible under the hood) we might see more. Small-T untouched (not a cluster code path). Correctness is structurally unchanged (same atomic barrier, same L2 coherence point, same PTX).

   **Confidence:** Medium. The Triton launch-plumbing for `num_ctas` + `cluster_dims` is documented in `triton/backends/nvidia/compiler.py:102-107,236-239` and `.../driver.py:318-358`; both kwargs flow through `backend.parse_options`. Risk: (i) clustering requires the grid divisibility we already satisfy (8 divides 8), (ii) we may hit a Triton compile warning or fallback if the backend refuses clustering for kernels with atomic ops — no evidence it does, but untested.

   **Revert trigger:** abs_err > 1.56e-02, kernel fails to compile, or A/B vs exp_18 is ≥ +1% mean on large T.

2. **Fallback (if Action 1 regresses or fails):** Small-T Q-load overlap. Concrete interleaving for `_fused_attn_kernel` (T≤2 path, `sparse_fused.py:196-292`):

   ```python
   # current order (sparse_fused.py:237-248):
   q_nope = tl.load(q_nope_ptrs)              # 17 KB, ~1.5 µs
   q_pe   = tl.load(q_pe_ptrs)                #  2 KB, ~0.3 µs
   m_i = tl.full(...); l_i = tl.zeros(...); acc = tl.zeros(...)
   offs_topk = tl.arange(0, TOPK)
   idx_scan = tl.load(Indices_ptr + t*stride_idx_t + offs_topk)   # 8 KB
   num_valid = tl.sum((idx_scan >= 0).to(tl.int32), axis=0)
   max_bn = ((num_valid + BLOCK_N - 1) // BLOCK_N) * BLOCK_N

   # proposed: issue idx_scan FIRST, then Q loads, then consume idx_scan.
   # Goal: overlap Q HBM bandwidth with int32 index scan arithmetic.
   offs_topk = tl.arange(0, TOPK)
   idx_scan = tl.load(Indices_ptr + t*stride_idx_t + offs_topk)   # issue first
   q_nope = tl.load(q_nope_ptrs)              # HBM can pipeline against scan compute
   q_pe   = tl.load(q_pe_ptrs)
   num_valid = tl.sum((idx_scan >= 0).to(tl.int32), axis=0)       # compute while Q arrives
   max_bn = ((num_valid + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
   m_i = tl.full(...); l_i = tl.zeros(...); acc = tl.zeros(...)
   ```

   **Why:** profile.md identifies Q-load as 1.9 µs (12%) of small-T latency and the indices-scan as an in-kernel cost hidden in "useful work". Triton will pipeline independent loads if they're issued before any dependent arithmetic; today the `tl.load(Indices_ptr ...)` is written *after* Q is consumed (`q_nope`, `q_pe` are ops waiting on load). Reordering makes indices hit HBM before Q, and the int32 reduce for `num_valid` runs while Q arrives.

   **Impact:** Projected −0.3 to −0.8 µs on T≤2 workloads (10.15 → ~9.5 µs), 9/23 workloads. Low risk — pure reordering, no algorithmic change.

   **Only run this if Action 1 regresses.** Do not couple them.

3. **Deferred (exp_21+):** Full cluster-barrier via inline PTX. Snippet to reserve (do NOT attempt this iteration):
   ```python
   # Replace the atomic-spin block with:
   tl.inline_asm_elementwise(
       "barrier.cluster.arrive.relaxed.aligned; barrier.cluster.wait.aligned;",
       constraints="=r",
       args=[],
       dtype=tl.int32, is_pure=False, pack=1,
   )
   ```
   Caveat: `inline_asm_elementwise` is documented as per-element in `triton/language/core.py:builder.create_inline_asm`; calling it with `args=[]` on an empty-shape output is an untested pattern in this codebase. If Triton rejects empty args, workaround is to pass a dummy int32 scalar tensor and return a dummy int32 result. Full validation of this path needs its own experiment; it is **not** a safe drop-in for the atomic-barrier today.

## Do not try
- **Full cluster-barrier PTX + cluster launch in one experiment** (exp_18 plan flagged this as coupled; still true for exp_20).
- **`num_warps ∈ {4, 16}` on the fused kernel** — exp_14, exp_17 both regressed (LESSON: num_warps=8 is strict optimum for H=16).
- **`num_stages=3`** — exp_19 regressed +6% (shmem overflow for 0-2 iter loop).
- **`num_stages=1`** — untested but the split loop has enough iters on large-T for stage-2 prefetching to matter.
- **bf16 partial_acc** — exp_12 regressed (L2-resident, no HBM saving).
- **Vectorized combine (3D load + axis-0 reduce)** — exp_16 regressed +19%.
- **NUM_SPLITS ≠ 8** — exp_4 regressed +33–50%; also couples with `D_CKV_SPLIT` and cluster size.
- **Caching `partial_m/l/acc` across calls** — I considered this (LESSON 8 — buffer persistence). Verified via `flashinfer.testing.bench_gpu_time_with_cupti` implementation at `flashinfer/testing/utils.py:1276-1279`: CUPTI measures `max_kernel_end − min_kernel_start` across the launched activities. `torch.empty` is purely host-side (no MEMSET, no MEMCPY), so allocator cost does NOT appear in the CUPTI span. Persisting these buffers would be code hygiene but not move measured latency. Skip.
- **Attacking the T=1 valid=2 outlier (5.12 µs/valid)** — it's launch-tax-floored at 10.24 µs, pre-compute compute is already ≤ 4 µs. Any sub-8 µs target requires persistent-kernel or CUDA-graph (forbidden). No angle.
- **`.item()` dispatch on `sparse_indices.max()` to route T=2 high-valid to split+combine** — the CPU-GPU round-trip is ~5 µs, eating the entire potential saving.

## Coordination notes

**Test plan:**
1. Modify launch site (3-line diff) on `solution/triton/sparse_fused.py`.
2. `modal run scripts/run_modal.py --quick` → correctness gate (T=1 + T=8). abs_err ≤ 1.56e-02.
3. `modal run scripts/ab_benchmark.py::run --a experiments/exp_18/sparse_fused.py` → paired A/B.
4. If A/B shows ≤ −0.3% mean on large T with ≥ 7/7 large-T wins: `/log-experiment`.
5. If A/B is tie/regress: revert the 3-line diff and pivot to Action 2 (small-T Q-load overlap) in the same exp_20 session.

**Speed:** target < 15 min total. Don't overspend investigating the cluster-launch mechanics on Modal; if it fails to compile, revert and run Action 2 immediately.

**Do NOT profile again before coding.** profile.md is 1 iteration old (exp_15 baseline) and still accurate for the overall phase breakdown — re-profiling adds ~10 min and won't change the plan.

**Signal interpretation:**
- Expected A/B win pattern (Action 1): −0.3 to −0.8% on 7 large-T workloads, ties on small-T.
- If wins are uniform ~0% across all large-T: cluster launch accepted but gave no scheduling benefit (CTAs already fit in one GPC at this small grid). Still a no-regression — keep it or revert per judgement.
- If large-T regresses >0.5%: cluster scheduling competing with other kernels on the device or shmem overflow. Revert.
