# Plan — exp 26

## Diagnosis
Plateau at ~0.016 ms large-T median (exp_18 → exp_25, ±0.5%). Scalar/PTX knobs are exhausted and exp_25 confirmed PDL is **structurally invisible** to CUPTI — any lever that only reduces cross-kernel wall-clock overlap cannot be rewarded. The remaining CUPTI-visible bucket is **split compute = 5.51 µs at T=8, 34% of total** (profile.md). With NUM_SPLITS=8 block-partitioned, split `s` owns TopK positions `[s*256 : (s+1)*256)`. Per workload_profile, p90 per-token valid=1089, p50=33, and valid is a contiguous prefix (contig p50=1.0). On the typical large-T token, **split 0 does ~1 BLOCK_N of work while splits 1-7 do 0** (after dynamic loop-bound cutoff); the fused kernel's split-phase duration equals the max-CTA time = split 0's work. Straggler imbalance is the dominant lever left.

## Strategy
**Targeted fix.** Change the split-phase work distribution from block-partition to **stride-partition**: split `s` owns TopK positions `{s, s + NUM_SPLITS, s + 2*NUM_SPLITS, ...}` so valid entries (which cluster at the TopK prefix) are distributed round-robin across all 8 splits. Partial tensors (`partial_m/l/acc[t,s,...]`) and combine phase are untouched — combine merges `NUM_SPLITS` partials regardless of which TopK positions each split consumed.

## Actions (priority ordered)

1. **What:** Rewrite the split-phase inner loop in `_fused_split_combine_kernel`:
   ```python
   # BEFORE:
   start = s * SPLIT_SIZE  # 256
   offs_split = tl.arange(0, SPLIT_SIZE)
   idx_scan = tl.load(Indices_ptr + t*stride_idx_t + start + offs_split)
   num_valid = tl.sum((idx_scan >= 0).to(tl.int32), axis=0)
   max_bn = ((num_valid + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
   for bn in range(0, max_bn, BLOCK_N):
       idx_ptrs = Indices_ptr + t*stride_idx_t + (start + bn + offs_n)
       ...
   ```
   to strided:
   ```python
   # AFTER:
   # Split s owns TopK positions {s, s+NUM_SPLITS, s+2*NUM_SPLITS, ...}
   offs_split = s + tl.arange(0, SPLIT_SIZE) * NUM_SPLITS  # stride-NUM_SPLITS scan
   idx_scan = tl.load(Indices_ptr + t*stride_idx_t + offs_split)
   num_valid = tl.sum((idx_scan >= 0).to(tl.int32), axis=0)
   max_bn = ((num_valid + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
   for bn in range(0, max_bn, BLOCK_N):
       topk_pos = s + (bn + offs_n) * NUM_SPLITS
       idx_ptrs = Indices_ptr + t*stride_idx_t + topk_pos
       ...
   ```
   Everything else (K/P gather, dots, online softmax, partial writes, atomic barrier, combine phase, small-T path) is **untouched**.

   **Why:** With valid=33 concentrated in TopK prefix, block-partition puts all work on split 0. Strided distribution spreads those 33 into splits 0..7 as {5, 4, 4, 4, 4, 4, 4, 4} → all 8 CTAs see ~1 BLOCK_N of work each in parallel → max-CTA time drops from ~5.5 µs to roughly 5.5/N where N is the number of splits that had work before (here N≈1–2), minus shared overhead. The `num_valid` pre-scan (LESSON for exp_6) still produces a tight dynamic bound per-split, so splits with 0 valid exit immediately without entering the K-load loop. Contiguity of K-cache rows is **not lost** because per-split K-row gather was already non-contiguous (sparse_indices values, not positions, drive cache rows) — and the workload's contig=1.0 means stride-8 TopK positions still map to nearby K-rows within one page (page_size=64 ≥ 8 × 8 stride). Cross-CTA page reuse is p50=1.0 so no L2 contention introduced.

   **Impact:** Projected **−1.0 to −2.0 µs on large T** (16.4 → 14.5–15.4 µs, −6 to −12%). Directly reduces per-kernel CUPTI duration by lowering max-CTA split time (the barrier-wait floor). Small T untouched.

2. **Safety / revert gates:**
   - **Correctness gate:** abs_err ≤ 1.56e-02 on `--quick` (strided distribution is a permutation of the same reduction; output is identical up to fp32-order associativity — expect bit-identical or within tolerance).
   - **Coalescing sanity:** if A/B shows a regression, check whether high-valid workloads (T=8, valid>2000) regress while low-valid win. If so, we've traded prefix-heavy win for dense-case K-bandwidth loss. Pivot to a hybrid: stride only when `num_valid_full_topk < threshold` (but requires a second pre-scan pass).
   - **A/B gate:** `scripts/ab_benchmark.py::run --a experiments/exp_24/sparse_fused.py`. Pass if ≥6/7 large-T wins with mean Δ ≤ -0.5%; revert otherwise.

3. **Test plan:** Quick-correctness first (`modal run scripts/run_modal.py --quick`), then stride-2 A/B vs exp_24. If A/B ambiguous, run twice on separate VMs for stability (per exp_18 precedent).

## Do not try
- **`launch_pdl=True` / inline-PTX `launch_dependents`** — CUPTI-invisible (LESSON 41, exp_25).
- **`num_ctas` / cluster_dims** — Triton 3.6 PlanCTA assertion on atomic-barrier kernels (LESSON 27, exp_20).
- **Gluon `gl.dot_fma`** — 80× software-FMA regression (LESSON 30, exp_22).
- **`num_warps` ≠ 8** — both directions regressed (LESSON 26, exp_14/17).
- **`num_stages` = 3 on fused_split_combine** — shmem overflow (exp_19).
- **NUM_SPLITS ≠ 8** — exp_4 +33–50%.
- **BLOCK_N retune / bf16 partial_acc** — exp_9/12/13 saturated.
- **`.cs` cache_modifier** — not supported on Triton 3.6 (LESSON 39).
- **3D vectorized combine / idx_scan reorder** — exp_16/20 no-op.

## Coordination notes
**One cleanly-scoped change** — 5-line edit to `_fused_split_combine_kernel`'s split phase; no host changes, no new launch kwargs, no coupled edits to combine or small-T kernel. Quick-correctness should bit-match (associativity aside). No profiling dependency — proceed directly. Budget 20 min for implementation + A/B. If the first A/B shows mixed signal, the fallback is trivial: either revert, or try a mid-point "first-prefix-balanced-then-strided" variant (more complex, save for exp_27).
