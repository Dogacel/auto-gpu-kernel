# Plan — exp 30

## Diagnosis

Three consecutive reverts (exp_27/28/29). Current best is exp_26 (stride-partition) at 0.016 ms large-T median. Profile breakdown (T=8): 30% launch / 12% Q-load / 14% barrier / 48% compute — every per-kernel lever is either saturated (cache modifiers exp_23/24/28, compact-block exp_27/29) or structurally blocked (PDL exp_21/25 invisible to CUPTI; cluster sync exp_20 incompatible with atomic barrier; num_warps/num_stages LESSON-26; Gluon exp_22 too slow without full Blackwell primitives). The root cause of the plateau is not a broken idea — it is that the remaining ~5 µs of large-T recoverable headroom (per profile §Concrete projections) all sit behind measurement or compiler walls we cannot traverse in Triton 3.6 under CUPTI.

The **one axis untouched** since exp_2 is the host-side Python/PyTorch overhead: `partial_m`, `partial_l`, `partial_acc` are re-allocated via `torch.empty(...)` on every call (sparse_fused.py:348-350). LESSON-22 established that `torch.zeros(...)` per call costs ~5 µs (hitting `cudaMemset` → visible in CUPTI), which is why `_counter_cache` persists the barrier counter. The analogous `torch.empty` path for the three scratch buffers has never been tested.

## Strategy

**Targeted structural move.** Extend the proven `_counter_cache` pattern (LESSON-22) to persist `partial_m`, `partial_l`, `partial_acc` across calls, keyed by `(device, num_tokens)`. This matches pathology checklist item 8 (buffer persistence) exactly, is low-risk (purely host-side change, no kernel edit, scratch buffers are write-before-read so no zero-init required), and tests a falsifiable hypothesis: "CUPTI sees some portion of `torch.empty` overhead, and caching removes it."

If this is a no-op (`torch.empty` from PyTorch's caching allocator truly never hits the CUDA runtime), we learn a durable fact and can close the axis. If it saves even 0.5 µs, that's applied to every single workload (23/23) with zero regression risk. Asymmetric payoff.

## Actions (priority ordered)

1. **What:** Add a module-level `_partial_buffer_cache: dict = {}` alongside `_counter_cache` (sparse_fused.py:299). Add a helper `_get_partial_buffers(num_tokens, device, NUM_SPLITS, H, D_ckv) -> (partial_m, partial_l, partial_acc)` that caches by `(device, num_tokens)` and reuses the three fp32 tensors. Replace sparse_fused.py:348-350 with a single call to the helper. Keep shapes and dtypes identical; buffers are written before read in every kernel call (all three partials are output of the split phase, then input to combine) so no zero-init is needed.

   **Why:** LESSON-22 measured ~5 µs regression from a single `torch.zeros(T, i32)` per call — that cost was all `cudaMemset`. `torch.empty` avoids the memset but still returns a fresh Tensor object: the caching allocator may still invoke `cudaMallocAsync` for new size keys, and even cache-hit paths incur a Python-object creation + refcount + stream-metadata-update. CUPTI captures cudaMalloc*/cudaFree* runtime calls. The three allocations here are 2.5 KB (partial_m), 2.5 KB (partial_l), 256 KB × num_tokens (partial_acc, the big one — 2 MB at T=8) per call. Across the 200-call inner loop of the benchmark harness, a cold allocator warms into cache-hits fast; but any size-key mismatch (e.g., the dispatcher switching between T=1, T=2, T=6, T=8 across workloads) forces fresh allocations. Caching by `(device, num_tokens)` eliminates that mismatch entirely.

   **Impact:** Best case ~1–3 µs saved per call (analogous to the counter cache's ~5 µs saving) applied to 23/23 workloads — median latency 0.016 → 0.013–0.015 ms on large T, proportionally more on small T (10 → 7–9 µs). Worst case zero effect (allocator already fully caches), zero regression risk. Net asymmetric.

2. **What (only if Action 1 is strictly zero):** Hoist `sm_scale * LOG2E` in the kernel wrapper to a local variable once (line 333, 366). Currently a Python float multiplication happens inside each launch-argument pack. Trivial but removes one CPython float op from the hot path.

   **Why:** Scalar host-side work during kernel argument marshalling is nominally free, but in a tight 200-call loop, 10-20 ns per multiplication × two kernels × 10k calls = micro-measurable in theory.

   **Impact:** Almost certainly below noise; keep as a secondary change only if the cache-buffer A/B is dead flat (neutral) so the iteration produces some signal.

## Do not try

- **Compact-block partition of any flavour.** Exp_27 (bug), exp_29 (fixed bug but +10% regression). Pre-scan of full 2048-entry TopK costs more than it saves. Axis closed.
- **Additional `cache_modifier` knobs.** Exp_23 (K loads `.cg`, kept marginal), exp_24 (partial stores `.cg`, kept marginal), exp_28 (combine loads `.cg`, reverted). Axis saturated — LESSON-40 explains the regime dependency.
- **`num_warps` sweep.** LESSON-26: strict optimum at 8 for H=16. Don't revisit.
- **`num_stages` sweep on large-T.** Exp_19 (=3) regressed. Split loop runs 1-2 iters — more stages means more shmem pressure with no overlap gain. The remaining untested value `num_stages=1` carries register-pressure risk; deferred unless this experiment creates signal that isolates prefetch overhead.
- **PDL / cluster sync / cooperative grid.** LESSON-27 (incompat with atomic barrier), LESSON-29 (no-op without upstream signal), LESSON-41 (structurally invisible to CUPTI). All three closed under current measurement methodology.
- **Gluon rewrite of the existing Triton path.** Exp_22 was 65-85× slower on `dot_fma`. A Blackwell-primitive rewrite (`bw.tcgen05_mma` + `bw.mbarrier`) would need 10+ iterations of runway and is high-risk given we're already within 1.5 µs of profile's 14.6 µs floor.
- **bf16 partial_acc.** Exp_12 reverted — partials are L2-resident (LESSON-18), so HBM-byte reduction has no effect.

## Coordination notes

- **Test on quick first** (2/2 correctness check), then A/B paired vs exp_26 via `scripts/ab_benchmark.py::run --a experiments/exp_26/sparse_fused.py`. Paired same-VM measurement is essential for sub-5% deltas (CLAUDE.md §One-optimization-per-iteration).
- **A/B expectation is bimodal**: either (a) consistent −0.2 to −1.0 µs on most workloads (B wins 10+/12, modest mean Δ negative) → new best, or (b) dead flat (B wins 5-7/12, mean Δ ≈ 0 ± 1 std) → close axis and pivot exp_31 to researching a Blackwell-primitive partial Gluon rewrite.
- **One change per iteration**: apply Action 1 only; hold Action 2 for next iteration if Action 1 is strictly zero.
- **Decision rule**: keep only if A/B mean Δ is clearly negative AND 8+/12 workloads favor B. A null result (4-7/12, |Δ| ≈ 0) is the expected alternative and should be documented cleanly to rule the axis out permanently.
