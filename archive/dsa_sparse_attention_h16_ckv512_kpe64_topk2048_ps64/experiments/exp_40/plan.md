# Plan — exp 40

## Diagnosis

We are 13 iterations past the last structural win (exp_26 stride-partition). Only exp_37 (monotonic counter, −2.88% on T≥3) landed a real improvement in that stretch. All cheap Triton scalar axes are closed: `num_stages`/`num_warps` (LESSON-26, exp_19/32/34), `NUM_SPLITS` (exp_4, exp_31), `BLOCK_N` (exp_33 shmem OOM), cache modifiers (exp_23/24/28, LESSON-40), compact-block partition (exp_27/29), buffer persistence (exp_30), D_CKV_SPLIT on fused (exp_36, LESSON-43), per-slot atomics (exp_38, LESSON-45), PDL (exp_21/25, LESSON-41), cluster (exp_20, LESSON-27). The combine-IO axis is closed by three independent experiments: exp_28 (.cg loads, regressed), exp_30 (buffer persistence, neutral), exp_39 (partial_m/l merge, regressed). The remaining recoverable buckets — barrier spin (~1.8 µs), Q-load replication (~1.92 µs), split MMA (~5.51 µs) — are **structurally blocked in Triton 3.6**: cluster planner asserts on mixed atomic+tiled ops (LESSON-27), `bw.mbarrier`/`bw.tcgen05_mma` are Gluon-only primitives. Per CLAUDE.md: "Gluon still counts as Triton." Research agent's Coordination Note in `experiments/exp_39/plan.md` is explicit: "exp_40 MUST adopt exp_35/plan.md's stage-1 Gluon pivot."

## Strategy

**Gluon stage-1 — replace `gl.dot_fma` with `bw.tcgen05_mma` in the split path.** This is the first of three staged Gluon migrations blueprinted in `experiments/exp_35/plan.md`. Stage 1 is explicitly an **investment iteration**: projected +5 to +20% regression vs exp_37 initially (2-launch tax + no atomic fusion yet), but unlocks stage 2 (`bw.mbarrier` re-fuse) and stage 3 (`num_ctas=8` cluster sync) which together target −9 to −11% vs exp_37. Per CLAUDE.md /optimize skill's 15–20 plateau threshold we are at iteration 13 — within mandate.

## Actions (priority ordered)

1. **Starting point:** `experiments/exp_22/sparse_fused.py`. This is the working Gluon split kernel using `gl.dot_fma` that passed 12/12 correctness at ~1.30 ms large-T (software-FMA floor, ~65–85× slower than Triton). The exp_22 scaffold already has:
   - All layouts tuned (`qn_layout`, `qp_layout`, `pn_layout`, `kc_layout`, `kp_layout`, `h_layout`)
   - Correct `DotOperandLayout(op, parent, k_width=0)` usage (LESSON-33)
   - Correct fp32-cast-before-convert_layout pattern (LESSON-34)
   - Online softmax with no atomic barrier (two launches: split → combine)
   - T≤2 fused kernel kept in Triton, unchanged

2. **Probe first:** Write `scripts/probe_gluon19.py` to validate `bw.tcgen05_mma` signature + layout constraints on a toy `[H=16, N=128, D=512]` shape. Discover: does it take positional args, what is the `use_acc` flag, does it need a separate commit/wait step, what TMEM descriptor shape is required, how does it compose with `gl.allocate_shared`? Do NOT edit `solution/triton/sparse_fused.py` until the probe compiles and returns a numerically correct MMA result on the toy shape.

3. **Port exp_22 scaffold → `solution/triton/sparse_fused.py`:**
   - Keep current `_fused_attn_kernel` for T≤2 (unchanged from exp_37)
   - Replace current `_fused_split_combine_kernel` with a two-launch pair: `_split_kernel_gluon` + `_combine_kernel`
   - For T≥3: compute `partial_m`, `partial_l`, `partial_acc` in Gluon split kernel; reduce in Triton combine kernel
   - Drop the atomic barrier (exp_22 pattern — re-introduce via `bw.mbarrier` in exp_41)

4. **Swap `gl.dot_fma` → `bw.tcgen05_mma`:** Inside `_split_kernel_gluon`, replace the two `gl.dot_fma(...)` calls for Q@Kc^T and Q@Kp^T. Target signature (discover actual via probe):
   ```python
   tmem_logits = bw.alloc_tmem([H, BLOCK_N], dtype=gl.float32)
   q_nope_smem = gl.allocate_shared([H, D_CKV], dtype=gl.bfloat16, layout=smem_layout_a)
   q_pe_smem   = gl.allocate_shared([H, D_KPE], dtype=gl.bfloat16, layout=smem_layout_b)
   # (within topk loop)
   kc_smem = gl.allocate_shared([BLOCK_N, D_CKV], dtype=gl.bfloat16, layout=smem_layout_k)
   kp_smem = gl.allocate_shared([BLOCK_N, D_KPE], dtype=gl.bfloat16, layout=smem_layout_k_pe)
   bw.tcgen05_mma(q_nope_smem, kc_smem, tmem_logits, use_acc=False)  # first dot
   bw.tcgen05_mma(q_pe_smem,   kp_smem, tmem_logits, use_acc=True )  # accumulate second
   bw.tcgen05_commit(tmem_logits)
   logits = bw.tmem_load(tmem_logits, layout=pn_layout)
   ```
   The softmax/partial-write tail stays as in exp_22.

## Acceptance gate

- Kernel compiles on Triton 3.7 / sm_100a
- `modal run scripts/run_modal.py --quick` passes **2/2** (correctness, both T=1 Triton path and T=8 Gluon path)
- Stride-2 large-T median **≤ 0.025 ms** (≈ 50% slower than exp_37 ~0.0155 ms). If this threshold is met, we're on track for stage 2; if latency ≥ 0.10 ms, MMA primitive isn't wired correctly — revert

## Do not try

See `experiments/exp_39/plan.md` "Do not try" section — all prior Triton dead-ends still apply. Plus:
- Do NOT try `bw.mbarrier` re-fuse in this iteration (that's stage 2 / exp_41)
- Do NOT try `num_ctas=8` cluster in this iteration (that's stage 3 / exp_42)
- Do NOT touch `_fused_attn_kernel` (T≤2 path) — out of scope for Gluon pivot

## Revert threshold

- Correctness fail on `--quick` → revert to exp_37 baseline
- Stride-2 large-T median > 0.025 ms → revert, declare stage 1 not-viable
- Stride-2 large-T median ≤ 0.025 ms → keep as scaffold for exp_41, even if not a new best

## Coordination notes

- **Investment iteration.** Not a new best. Log clearly as "scaffold for exp_41/42" in result.md.
- **Time budget:** if probe_gluon19 doesn't land cleanly in 3 Modal iterations, fallback is to inspect `scripts/probe_gluon18.py` + `experiments/exp_22/sparse_fused.py` for adjacent API patterns.
- **If bw.tcgen05_mma API is different than expected** (e.g. needs `bw.tcgen05_mma_commit` as separate call, or different TMEM allocation pattern), adjust based on probe findings — the plan sketches the expected signature but the exact form is discovered via probe.
