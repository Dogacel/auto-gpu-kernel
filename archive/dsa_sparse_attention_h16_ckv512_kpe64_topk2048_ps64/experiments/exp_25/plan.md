# Plan — exp 25

## Diagnosis
Local minimum at exp_18 (~0.016 ms large-T median). Last 6 attempts (exp_19-24) all landed within ±0.5% of exp_18, below VM noise. Scalar Triton knobs (`num_warps`, `num_stages`, `num_ctas`, `launch_pdl` alone) and cache modifiers (`.cg`) are saturated. The one lever LESSONS-29 explicitly flags as un-closed: `launch_pdl=True` worked but was a **no-op** in exp_21 because Triton does not auto-emit `griddepcontrol.launch_dependents` PTX at kernel tail — so the prior launch never signals the next.

## Strategy
**Targeted fix.** Close the exp_21 PDL loop by emitting the missing upstream `griddepcontrol.launch_dependents` signal via `tl.inline_asm_elementwise` at the kernel epilogue, and re-enable `launch_pdl=True` on the launch. One coordinated, coupled change — both pieces are *required* to get any signal.

## Actions (priority ordered)

1. **What:** Add at the tail of both `_fused_split_combine_kernel` (after `atomic_add(-1)`) and `_fused_attn_kernel` (after `tl.store(out_ptrs, ...)` / `lse` store):
   ```python
   tl.inline_asm_elementwise(
       "griddepcontrol.launch_dependents;",
       "=r", [], dtype=tl.int32, is_pure=False, pack=1)
   ```
   Re-add `launch_pdl=True` to BOTH launch sites (reverted in exp_21).
   **Why:** PDL semantics: each iteration's kernel signals *the next queued kernel* (which is another launch of ourselves in the benchmark hot loop) to begin its prologue — constant-bank fetch, SM dispatch setup — while our current tail still executes. This overlaps ~1 µs of driver/SM dispatch with tail work. Applies to every launch in the ~200-call hot loop. Value comes from hiding part of the 4.94 µs noop floor per launch.
   **Impact:** Projected −0.5 to −1.5 µs per call on both T≤2 and T≥3 paths (small-T 10.2 → 9.3-9.7 µs; large-T 16.4 → 15.3-15.9 µs, −3 to −9%). LESSON-29 says "~1 µs of driver setup overlap" — take the midpoint as −0.8 µs expected. Recovers driver-dispatch overlap but NOT the 5 µs hardware SM-dispatch floor (that is physical, not driveable).

2. **Safety / revert triggers:**
   - If `tl.inline_asm_elementwise` rejects `args=[]` empty-input (untested per exp_20 plan lines 83), pass a dummy int32 tensor and discard the return. Have fallback ready.
   - Verify `"griddepcontrol.launch_dependents;"` assembles on sm_100a (B200). If PTX assembler rejects (wrong SM target), try `"griddepcontrol.launch_dependents;\n"` with `".version 8.0"` hint in a secondary probe.
   - Correctness gate: abs_err ≤ 1.56e-02 (unchanged — PDL is a launch-ordering change, cannot perturb numerics).
   - A/B gate: if mean Δ ≥ 0 with no large-T wins consistent-direction, revert both the inline-asm and the kwarg.

3. **Test plan:** `--quick` correctness, then `scripts/ab_benchmark.py::run --a experiments/exp_24/sparse_fused.py` paired A/B. Targeted: ≥ 6/7 large-T wins, mean Δ ≤ -0.3%.

## Do not try
- **`num_ctas` / `cluster_dims` / cluster-barrier PTX** — exp_20 confirmed Triton 3.6 `PlanCTA.cpp` assertion blocks this on atomic-barrier kernels.
- **`launch_pdl=True` alone** — exp_21 no-op without the PTX signal.
- **Gluon `gl.dot_fma`** — exp_22 80× regression (LESSON 30, software FMA).
- **`num_warps` ≠ 8** — exp_14/17 both regressed, LESSON 26 marks H=16 pinned.
- **`num_stages=3` on fused_split_combine** — exp_19 shmem overflow.
- **NUM_SPLITS ≠ 8** — exp_4 +33-50%.
- **BLOCK_N retune** — exp_9/13 saturated.
- **bf16 partial_acc** — exp_12 L2-resident, no HBM saving.
- **`.cs` cache_modifier** — LESSON 39 unsupported in Triton 3.6.
- **3D vectorized combine / `idx_scan` reorder** — exp_16/20 no-op.

## Coordination notes
One coupled change but both pieces are intrinsic to making PDL work — this is not two independent levers being bundled. Quick iteration: single-file 2-line edits + launch-kwarg. If the inline-asm syntax errors on first compile, spend ≤5 min probing `tl.inline_asm_elementwise` with a dummy return-tensor pattern; if still failing, pivot to Action 2 fallback (reverse the kernel-tail pattern: `griddepcontrol.wait;` at kernel prologue to delay next-launch until ours finishes — inverse semantics, but tests the same PTX path). Budget 15 min total.
