# Plan — exp 39

## Diagnosis

We are 12 experiments (exp_27..exp_38) past the last structural win (exp_26 stride-partition) with only exp_37 (monotonic counter, −2.88% on T≥3) landing a real improvement. All cheap scalar Triton axes are now closed by prior experiments: `num_stages`/`num_warps` (LESSON-26, exp_19, exp_32, exp_34), `NUM_SPLITS` (exp_31 −62%), `BLOCK_N=256` (exp_33 shmem OOM), cache modifiers in both directions (exp_23/24/28, LESSON-40), compact-block partition (exp_27/29), buffer persistence (exp_30), `D_CKV_SPLIT=1` on fused (exp_36, LESSON-43), per-slot atomics (exp_38, LESSON-45), and all scalar `launch_pdl`/`cluster_dims` kwargs (LESSON-27, LESSON-29, LESSON-41). Profile.md (from exp_15) is also stale post-exp_26/exp_37 — the 2.23 µs barrier-spin bucket it names has shrunk by ~0.44 µs and the split-compute distribution shifted with stride-partition. The kernel is ~1 µs (6%) away from profile.md's projected Triton-only floor (~14.6–14.9 µs), with the remaining recoverable bucket (the ~1.8 µs spin) requiring structural change — cluster-sync or `bw.mbarrier` (exp_35 Gluon plan).

## Strategy

**Targeted fixes.** Test the last remaining micro-axis in the combine phase: merge `partial_m` and `partial_l` into a single interleaved `[num_tokens, NUM_SPLITS, 2*H]` tensor so the combine CTA issues one fused scalar load per split instead of two. This is a clean, revertable, single-axis change. If it wins, great; if it ties, the combine-IO axis is closed and we commit to the Gluon pivot (exp_35/plan.md) in exp_40 with full justification. Budget: this is **iteration 13 of the 15–20 threshold** — exp_40 is the Gluon go/no-go decision regardless of exp_39's outcome.

## Actions (priority ordered)

1. **What:** In `solution/triton/sparse_fused.py`, merge `partial_m [T,S,H]` and `partial_l [T,S,H]` into `partial_ml [T,S,2H]` (interleaved: `[:,:,0:H]=m, [:,:,H:2H]=l`). Modify `_fused_split_combine_kernel` so:
   - Lines 115–116: compute a single `pml_ptrs = Partial_ml_ptr + t*... + s*... + tl.arange(0, 2*H)` and issue one scalar store (2H=32 f32s) concatenating `m_i` and `l_i` (via `tl.cat` or `tl.join`, or by explicit offset arithmetic).
   - Lines 149–152 (combine phase inner loop): replace the two loads (`m_si = tl.load(pm_ptr_s)`, `l_si = tl.load(pl_ptr_s)`) with a single load of `[2H]` f32, then split into `m_si = ml[:H]` / `l_si = ml[H:]` via `tl.reshape` or slicing.
   - Host-side: replace the two `torch.empty((num_tokens, NUM_SPLITS, H), ...)` allocations with one `(num_tokens, NUM_SPLITS, 2*H)`, pass a single pointer and single set of strides to the kernel, drop the `Partial_l_ptr` kwarg.

   **Why:** The combine phase's `tl.static_range(NUM_SPLITS=8)` unrolled loop currently issues **16 small scalar loads** per CTA (8 splits × 2 tensors at 16 × 4 = 64 B each). These serialize through the L2 load-return path. Merging them into 8 single loads of 128 B each halves the in-flight load count and doubles per-load bytes (better L2 utilization). Each scalar load has a ~3–5 ns issue cost + L2 latency; halving the count saves ~25–40 ns per CTA × 8 unrolled iters × 8 CTAs = up to ~200 ns wall-clock on the critical-path CTA. Exp_38's discovery #3 established that per-iter cost in the spin loop at this µs scale is measurable. Same principle applies here. Writes on the split-phase side also collapse 2→1 (lines 124–125 vs line 126), halving the L2 RMW transactions hitting the partial region.

   **Impact:** Projected **−0.2% to −1.0% on T≥3 workloads** (~0.05–0.15 µs on the current 16 µs large-T baseline). Small-T fused path (T≤2) is **not touched** — same `_fused_attn_kernel`. If the saving is below noise (<0.3%), we declare the combine-IO axis closed.

2. **What:** If action 1 ties or regresses, run **one confirmation A/B** with `scripts/ab_benchmark.py` on a fresh VM before reverting. This is belt-and-suspenders: exp_38's first run showed 3/12 wins / mean=+0.0000 ms (looked like a tie), but run 2 exposed a consistent +0.5–1.2% regression. Same-VM paired A/B is the only reliable signal for sub-1% deltas.

   **Why:** sub-1% deltas vs previous best require paired-VM measurement (CLAUDE.md rule). A single stride-2 run is insufficient to distinguish "tied" from "mild regression" in this regime.

   **Impact:** Validates the outcome cheaply (~2 min × 2 runs).

3. **What:** Regardless of exp_39 outcome, the follow-up experiment is **scheduled as the Gluon go/no-go**. If exp_39 wins, we've recovered ~0.1 µs and have budget for one more Triton axis before Gluon (~iter 14/20). If exp_39 ties/regresses, all Triton levers on the T≥3 path are exhausted; **exp_40 MUST adopt exp_35/plan.md's stage-1 Gluon pivot** (bring up `bw.tcgen05_mma` in the split path, accept a +5–20% regression for one iteration as explicit investment). Record this decision as a Coordination Note so the optimizer doesn't cycle back into the Triton hunt.

   **Why:** CLAUDE.md's /optimize skill sets the Gluon threshold at 15–20 plateau iterations. exp_39 is #13. Without a clear trigger, we risk burning 5+ more iterations on sub-1% Triton micro-tuning when the 1.8 µs spin requires `bw.mbarrier` / cluster sync to attack.

## Do not try

- **`NUM_SPLITS ≠ 8`** in any direction — exp_4 (=16 regresses), exp_31 (=4 regresses −62%). LESSON-12.
- **`num_stages ≠ 2`** on either kernel — exp_19 (split=3 regress), exp_32 (split=1 regress), exp_34 (fused=3 regress). All directions closed.
- **`num_warps ≠ 8`** on H=16 — exp_14 (=4 regress), exp_17 (=16 regress). LESSON-26 locked.
- **`BLOCK_N=256`** on split — shmem OOM (exp_33). `num_stages=1` would fit but regresses (exp_32).
- **`cache_modifier=".cg"` on combine-phase LOADS** — exp_28 regressed; L1 tag-check overlapped with MMA compute. LESSON-40.
- **Compact-block partition** (exp_27, exp_29) — full-TopK pre-scan overhead exceeds coalescing benefit. Axis closed.
- **`torch.empty` persistence across calls** (exp_30) — PyTorch caching allocator already masks the cost; helper-fn overhead regresses.
- **`D_CKV_SPLIT_FUSED=1`** on T≤2 path (exp_36) — D-parallelism in the fused kernel is load-bearing, not waste. LESSON-43.
- **Per-slot ready-vector barrier** (exp_38) — same-cache-line atomics don't parallelize on B200. LESSON-45.
- **`launch_pdl=True` / `griddepcontrol.launch_dependents`** (exp_21, exp_25) — structurally invisible to CUPTI. LESSON-41.
- **`num_ctas=N` cluster on current kernel** (exp_20) — `TritonGPUPlanCTAPass` asserts on mixed atomic + tiled ops. LESSON-27. Gluon-only primitive.
- **Don't "fix" line 267's redundant `kc_slice` load in `_fused_attn_kernel`** — the second load is L1-hot (populated by line 251's kc load in the same iter), cost is ~5 ns; fixing it requires either 8× redundant compute (LESSON-43 regression regime) or 8× launch tax (compile-time d specialization). Both closed.

## Revert threshold

- **Stride-2 A/B vs exp_37 (previous best):** B wins ≥7/12 workloads AND mean Δ ≤ −0.0001 ms → kept. Either threshold failing → revert.
- **Second confirmation A/B on a fresh VM:** same gates. A single run can't distinguish tie from mild regression (exp_38's two-run discipline applies here).
- **Hard revert at +1% regression or worse** on any T≥3 workload, no second run needed — that's clear signal.

## Fallback if it fails

**exp_40 is the Gluon go/no-go.** Restart from `experiments/exp_35/plan.md` (preserved Gluon pivot blueprint) with the explicit understanding that stage 1 (replace `gl.dot_fma` with `bw.tcgen05_mma` in the split path) is an investment iteration projected at +5–20% regression vs exp_37. This violates the "one-iteration revert on regression" rule of the /optimize skill per exp_35's own abandoned-note — but the skill's 15–20 plateau threshold is its override. At exp_40 we'll be at iteration 14, within the mandate. The stage 1 plan:

1. Write `scripts/probe_gluon19.py` to validate `bw.tcgen05_mma` signature + layout constraints on a toy `[16, 128, 512]` shape.
2. Port exp_22's Gluon split scaffold (`experiments/exp_22/sparse_fused.py`) into `solution/triton/sparse_fused.py`, replacing `_fused_split_combine_kernel` for T≥3. Keep `_fused_attn_kernel` (T≤2) in Triton, unchanged.
3. Swap the two `gl.dot_fma(...)` calls for a single `bw.tcgen05_mma` path using tensor-memory descriptors (`bw.alloc_tmem`) and shmem staging (`gl.allocate_shared`).
4. Acceptance gate: quick correctness pass + stride-2 large-T median ≤ 0.025 ms (50% slower than exp_37 is acceptable; ≥ 0.10 ms means MMA not wired correctly — revert to exp_37 and declare Gluon axis closed).

If Gluon brings up cleanly, exp_41 does stage 2 (`bw.mbarrier` replace atomic barrier) targeting match/−2% vs exp_37. Exp_42 does stage 3 (`num_ctas=8` cluster sync) targeting −9 to −11% final.

If exp_40 can't land the probe cleanly in one iteration (wrong API surface, layout incompatibility, etc.), fallback-to-fallback is the **D-parallel warp-split on fused T≤2 kernel** direction from exp_35 plan's "2nd-best" section — a pure-Triton in-CTA warp-specialization of `_fused_attn_kernel` targeting the T=1/T=2 noop floor. Never tried. Expected −0.5 to −1.5 µs on T≤2. Risk: Triton doesn't cleanly expose warp IDs; may require `tl.inline_asm` for warp specialisation.

## Coordination notes

- **Quick iterations.** Action 1 is a ~10-minute edit + two ~2-min A/B runs. No re-profiling needed for this change — the projected saving is too small to shift the bucket picture materially. Re-profile at exp_40 if the Gluon path opens up, to measure the mbarrier/cluster recovery cleanly.
- **No reference impl to read first.** The change is a simple tensor-layout refactor. Test via `--quick` for correctness (abs_err should be identical — partial_ml is just reshaped storage, semantics unchanged), then stride-2 A/B for latency.
- **Do NOT couple any other change into exp_39.** Previous experiments occasionally bundled a "cheap" secondary change (exp_20 Action 2) that then confounded the A/B signal. One axis, one iteration — strict.
- **If A/B run 1 shows a clean 10+/12 win with mean Δ ≤ −0.0002 ms, skip run 2.** That's a confident win; no need to re-verify.
- **If the exp_39 kernel fails to compile** (e.g., `tl.cat` / `tl.join` API mismatch in Triton 3.6), fall back to writing two adjacent scalars via pointer arithmetic:
  ```python
  pml_ptr_base = Partial_ml_ptr + t*stride_pml_t + s*stride_pml_s
  tl.store(pml_ptr_base + offs_h, m_i)           # [0:H]
  tl.store(pml_ptr_base + H + offs_h, l_i)       # [H:2H]
  # Combine side: single load of [0:2H], split via tl.reshape([2, H])
  pml_ptr_s_base = Partial_ml_ptr + t*stride_pml_t + si*stride_pml_s
  ml_both = tl.load(pml_ptr_s_base + tl.arange(0, 2*H))
  ml_both_2d = tl.reshape(ml_both, [2, H])
  m_si = ml_both_2d[0]
  l_si = ml_both_2d[1]
  ```
  This still produces one fused load on the combine side even if the store side remains two calls (writes are not the critical-path direction).
