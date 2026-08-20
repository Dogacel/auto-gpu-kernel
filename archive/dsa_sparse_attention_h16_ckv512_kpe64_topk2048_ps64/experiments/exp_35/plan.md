abandoned: Premature — we're at 8 reverts since exp_26, not the 15-20 threshold the CLAUDE.md skill requires before committing to the Gluon pivot. The plan itself projects +5-20% regression on stage 1, conflicting with the "revert on regression" rule that must hold per-iteration. Preserve the plan as a reference for when Triton is truly exhausted (keep the 3-stage migration blueprint, the `bw.tcgen05_mma` sketch, and the do-not-try list). One untested Triton-side axis remains: **collapse D_CKV_SPLIT on the fused kernel from 8 to 1** to eliminate 7/8 redundant Q@K^T + softmax compute replication on launch-bound T=1/T=2 workloads (exploiting LESSON-16's observation that D-parallel in the fused path replicates that compute). Moving to exp_36 for that attempt; revisit this plan if exp_36-40 all fail.

# Experiment 35 Plan — Gluon pivot, stage 1: replace `gl.dot_fma` with `bw.tcgen05_mma` in the split phase

## Hypothesis

All Triton-feasible levers are exhausted. Eight consecutive reverts since exp_26 touched every remaining scalar knob — `NUM_SPLITS` (exp_31 -62%), `num_stages` on both kernels (exp_19, exp_32, exp_34), `BLOCK_N=256` (exp_33 shmem OOM), cache modifiers in all directions (exp_23/24/28), host-side caching (exp_30), and compact-block partition (exp_27/29). `profile.md` projects the single remaining recoverable bucket as the **2.23 µs atomic-barrier spin on large T** (−9 to −11% latency). Eliminating it requires cluster-sync or warp-specialization — both of which are **structurally blocked in Triton 3.6** per LESSON-27 (cluster planner asserts on mixed atomic + tiled ops) and un-exposed in its frontend. The only reachable primitive set that gives us (a) tensor-core MMA with explicit shmem control, (b) `bw.mbarrier` for in-CTA sync, and (c) `num_ctas` clusters with `barrier.cluster.arrive/wait` is the Blackwell intrinsics inside Gluon, which CLAUDE.md treats as Triton ("Gluon still counts as Triton").

Exp_22 already did the correctness bring-up for a Gluon split kernel using `gl.dot_fma` (software FMA, ~65–85× slower than tensor-core `tl.dot`). Swapping that for `bw.tcgen05_mma` is the **one concrete change** that closes the perf gap vs the Triton baseline and unlocks the subsequent structural levers (stages 2 and 3 below). Per LESSON-30, `dot_fma` was always a scaffold, not a perf path; `tcgen05_mma` is the Blackwell tensor-core primitive.

## Current state

`solution/triton/sparse_fused.py` is exp_26's Triton-only code — hybrid dispatch (fused for T≤2, split+combine-via-atomic-barrier for T≥3) with stride-partition TopK distribution, `NUM_SPLITS=D_CKV_SPLIT=8`, `BLOCK_N=128`, `num_stages=2`, `num_warps=8`. Baseline is **0.016 ms large-T median / 0.0120 ms aggregate** from exp_26. The exp_22 Gluon scaffold at `experiments/exp_22/sparse_fused.py` lines 22–230 contains a two-launch Gluon split kernel + a plain-Triton combine kernel that compiles and passes 12/12 correctness but runs at **~1.30 ms large-T** — the `gl.dot_fma` overhead. That scaffold is the starting point for this experiment.

## Change

**Goal:** rewrite the split path of the exp_22 Gluon kernel to use `bw.tcgen05_mma` for both `Q@K_ckv^T` and `Q@K_pe^T` dots. Keep the T≤2 Triton fused kernel untouched. Keep the combine kernel plain Triton (same as exp_22 — cheaper to keep working than re-port).

### File layout (single-file; no staged migration needed yet)

Edit `solution/triton/sparse_fused.py` to:

1. **Copy the exp_22 Gluon split kernel into `solution/triton/sparse_fused.py`** as `_split_kernel_gluon` (replacing the current `_fused_split_combine_kernel` for T≥3 workloads). Restore the two-launch pattern (split → combine) — we lose the atomic barrier's 2 µs but gain the path to cluster sync in stage 2.
2. **Keep exp_26's Triton `_fused_attn_kernel`** for T≤2 unchanged.
3. **Port a D-parallel Triton combine kernel** (copied from exp_22's `_combine_kernel`, or equivalently the combine portion of exp_26's fused kernel).
4. **Inside `_split_kernel_gluon`**, replace the two `gl.dot_fma(...)` calls with a single `bw.tcgen05_mma` path using tensor memory:

```python
# Concrete change (within the inner split loop):
# Before (exp_22, lines ~162-180 in exp_22/sparse_fused.py):
#   q_nope_f32 = q_nope.to(gl.float32)
#   kc_t_f32 = kc_t.to(gl.float32)
#   q_nope_dot = gl.convert_layout(q_nope_f32, dot_q_pn)
#   kc_t_dot = gl.convert_layout(kc_t_f32, dot_k_pn)
#   logits = gl.dot_fma(q_nope_dot, kc_t_dot, acc_zero)
#   ... (same for kp dot) ...
#
# After:
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import blackwell as bw

# At kernel start: allocate tensor memory descriptors for the two dots.
# TMEM holds the fp32 accumulators for tcgen05_mma.
# Allocate ONCE per CTA; reuse across the topk loop via mbarrier stages.
tmem_logits = bw.alloc_tmem([H, BLOCK_N], dtype=gl.float32)  # fp32 acc for Q@Kc^T + Q@Kp^T

# Q tiles: stage into shared memory once (reused across topk iters via K-reload alone).
q_nope_smem = gl.allocate_shared([H, D_CKV], dtype=gl.bfloat16, layout=smem_layout_a)
q_pe_smem   = gl.allocate_shared([H, D_KPE], dtype=gl.bfloat16, layout=smem_layout_b)
gl.memcpy_async(q_nope_smem, q_nope_tile)
gl.memcpy_async(q_pe_smem,   q_pe_tile)
gl.memcpy_async_commit_group()
gl.memcpy_async_wait_group(0)

# Inside topk loop (per BLOCK_N tile):
kc_smem = gl.allocate_shared([BLOCK_N, D_CKV], dtype=gl.bfloat16, layout=smem_layout_k)
kp_smem = gl.allocate_shared([BLOCK_N, D_KPE], dtype=gl.bfloat16, layout=smem_layout_k_pe)
gl.memcpy_async(kc_smem, kc_gmem)
gl.memcpy_async(kp_smem, kp_gmem)
gl.memcpy_async_commit_group()
gl.memcpy_async_wait_group(0)

# Dispatch tcgen05 MMA: Q @ Kc^T   (accumulates into tmem_logits, clear on first)
bw.tcgen05_mma(q_nope_smem, kc_smem, tmem_logits, use_acc=False)  # first dot
bw.tcgen05_mma(q_pe_smem,   kp_smem, tmem_logits, use_acc=True )  # accumulate second dot

# Wait for MMA completion before reading logits to registers
bw.tcgen05_commit(tmem_logits)
logits = bw.tmem_load(tmem_logits, layout=pn_layout)  # [H, BLOCK_N] fp32
```

If `bw.tcgen05_mma`'s exact signature differs from above (it may be positional with a dedicated `use_acc` flag or it may use a separate `tcgen05_mma_commit`), discover the correct form via a probe script `scripts/probe_gluon19.py` *before* editing `solution/triton/sparse_fused.py`. LESSON-33 and LESSON-35 cover the layout + barrier discovery pattern.

Rest of the split loop (softmax update, partial_m/l/acc writes) stays as in exp_22.

### Acceptance gate

- Kernel compiles on Triton 3.7 / sm_100a.
- `modal run scripts/run_modal.py --quick` passes 2/2 (correctness, both T=1 via Triton path and T=8 via Gluon path).
- Large-T latency ≤ 0.025 ms (≈ 50% slower than exp_26). If this threshold is met, we're on track for stage 2 (re-add atomic/mbarrier fusion). If latency is ≥ 0.10 ms, the MMA primitive isn't wired correctly — fall back to exp_22 Gluon scaffold.

## Expected outcome

**Large T (stride-2 A/B vs exp_26):**
- If `bw.tcgen05_mma` lands cleanly: **large-T latency 0.017–0.020 ms**, 5–20% slower than exp_26 *initially* (cost of two launches + no atomic fusion yet). Not a win. **Acceptable ceiling** — stage 2's atomic/mbarrier re-fusion recovers the 2-launch tax; stage 3's cluster sync recovers the 2.2 µs barrier spin. Target end-of-stage-3: **0.014 ms** (−12% vs exp_26).
- If wiring fails: large-T stays at 1.3 ms (exp_22 software-FMA floor). Revert within iteration.

**Small T (T≤2):** unchanged. This experiment doesn't touch `_fused_attn_kernel`.

**Per-stage µs projection (if full 3-stage migration lands):**

| Stage | Large-T large-T latency | Δ vs exp_26 | Mechanism |
|---|---|---|---|
| 35 (MMA, 2-launch) | 0.017–0.020 ms | +5 to +20% | Gluon reaches tensor-core parity; +2 µs launch tax not yet recovered |
| 36 (mbarrier re-fuse) | 0.016 ms | 0 (match) | `bw.mbarrier` replaces atomic_add barrier; recovers 2 µs |
| 37 (cluster sync) | 0.0145 ms | −9% | `num_ctas=8` cluster + `barrier.cluster.arrive/wait`; 2.2 µs spin → 0.5 µs |

Stage 35 alone is not a win. **This plan is an investment**: we accept a neutral-to-slight-regression iteration to unblock the 9% large-T headroom behind cluster sync. This is explicitly what the mandate authorizes ("If you genuinely believe Triton is exhausted, explicitly recommend the Gluon rewrite path").

## How to validate

1. **Compile check first** via `scripts/probe_gluon19.py` (new). Probe `bw.tcgen05_mma` signature against `gl.allocate_shared` / `bw.alloc_tmem`. Do NOT edit `solution/triton/sparse_fused.py` until probe compiles and matches the expected pattern on a toy shape `[16, 128, 512]`.
2. **Correctness:** `modal run scripts/run_modal.py --quick` (2 workloads, ~90 s). Must pass 2/2 with `abs_err ≤ 1e-2`. If fails, compare against exp_22 scaffold (known-correct at `gl.dot_fma`) — the MMA swap is the only variable.
3. **Latency:** `modal run scripts/run_modal.py --stride 2` (12 workloads, ~2 min). The acceptance gate is **large-T median ≤ 0.025 ms**. This is 50% slower than exp_26 but ~65× faster than exp_22 — it proves the MMA primitive is live.
4. **A/B vs exp_26** is **not expected to win this iteration.** Skip the `ab_benchmark.py` paired run. Log the Gluon result, explicitly document "kept as scaffold, not a new best, pivot to stage 2 in exp_36."

## Risks / fallback

**Primary risk: Gluon API surface.** `bw.tcgen05_mma` may require `bw.alloc_tmem` calls that don't compose with Gluon 3.7's `gl.allocate_shared` layouts cleanly. If probes reveal blockers (e.g. TMEM descriptor mismatch, async commit semantics), the stage-1 fallback is:

**2nd-best direction — D-parallel warp split on fused kernel (Triton 3.6, no Gluon):** rewrite `_fused_attn_kernel` (T≤2 path) so the 8 warps within one CTA own 8 slices of TopK each, using `tl.where`-masked partial accumulation inside each warp and a shmem-based final reduction. This is structurally similar to flash-decoding but all within one CTA, eliminating the D-split grid replication that bloats T=1 compute. Grid becomes `(T,)` = 1 CTA per token on T=1 → direct attack on the 5.15 µs noop floor. Expected saving: 0.5–1.5 µs on T=1/T=2. Unexplored in Triton. Risk: Triton doesn't expose warp IDs cleanly; may require `tl.inline_asm` for `warp.idx`. This is the next experiment if exp_35 Gluon probe fails to compile.

**3rd-best direction — BLOCK_N_FUSED=32 on T≤2 path.** LESSON-19 predicts this wins on `valid < 32` tokens (a minority) and regresses on `valid > 200`. Low ceiling; same shape as exp_13's BLOCK_N_FUSED=64 win. Only worth trying if both Gluon and warp-split stall.

## Do-not-try (dead ends from prior experiments)

- `NUM_SPLITS ≠ 8` in any direction (exp_4: 16 regresses; exp_31: 4 regresses −62%).
- `num_stages ≠ 2` on either kernel (exp_19 split =3, exp_32 split =1, exp_34 fused =3 — all regress).
- `num_warps ≠ 8` on H=16 kernels (exp_14 =4 regresses; exp_17 =16 regresses). **Locked by LESSON-26.**
- `BLOCK_N=256` on split — shmem OOM with `num_stages=2` (exp_33). Would require `num_stages=1` first, which regresses.
- `cache_modifier=".cg"` on combine-phase loads (exp_28 regresses); on fused-kernel K loads (LESSON-40 regime-dependent).
- Compact-block partition (exp_27/29) — full-TopK pre-scan cost exceeds coalescing benefit.
- Persistent buffer cache (exp_30) — PyTorch caching allocator already masks `torch.empty` cost, and helper fn overhead regressed.
- `launch_pdl=True` / `griddepcontrol` PTX (exp_21, exp_25) — invisible to CUPTI per LESSON-41.
- `num_ctas=N` cluster directly on the current atomic-barrier fused kernel (exp_20 Action 1) — `TritonGPUPlanCTAPass` asserts per LESSON-27. **Only possible inside Gluon or after atomic barrier is removed.**
- `tl.debug_barrier`, `launch_cooperative_grid` scalar-kwarg-only tries — LESSON-29 "Triton scalar kwargs now fully exhausted."

## Coordination notes

**Quick iterations preferred** this time. The Gluon probe loop — probe_gluon19.py → quick-mode correctness → stride-2 latency — is fast (~3 minutes per cycle). Do NOT attempt stage 2 (mbarrier) and stage 3 (cluster) in the same iteration. One primitive per experiment, per CLAUDE.md rule "One optimization per iteration."

**Profile after stage 3 (exp_37) is landed**, not before — the current profile.md is stale w.r.t. Gluon. Re-profile will tell us whether cluster sync actually collapsed the barrier bucket to ~0.4 µs as predicted, or whether some other cost emerged.

**Reference impl to read before coding:** `experiments/exp_22/sparse_fused.py` (the working Gluon-split scaffold with all layouts tuned), `experiments/exp_26/sparse_fused.py` (the Triton baseline to match), and `scripts/probe_gluon18.py` (the most recent successful probe — examples of `gl.convert_layout`, `gl.arange`-with-layout patterns, and what the LESSON-33/34/35/36 constraints look like in practice).

**Time budget:** if 3 consecutive Gluon iterations (exp_35, 36, 37) can't land below 0.016 ms, revert `solution/triton/sparse_fused.py` to exp_26 and declare the Gluon axis closed. This is the commit from exp_22's conclusion: "Kept for follow-up; revert if Gluon can't beat exp_18 within a few iterations." We're now 13 experiments past that checkpoint — the Gluon path gets 3 more iterations before closure.
