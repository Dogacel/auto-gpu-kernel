# Plan — exp 55

## Diagnosis

Plateau at exp_51 (NS=16, new best). Last 3 experiments (exp_52 dynamic range, exp_53 combine rewrites, exp_54 adaptive dispatch) all reverted — combine-loop reformulation is closed (LESSON-55/56) and host-side `.item()` dispatch is catastrophic (LESSON-57, 9× regression). The only truly re-open knob post-exp_51 structural change is `num_stages` on the split loop: **under NS=16 with stride-partition, every T≥3 workload runs a 1-iter split loop** (per-split valid ≤ ceil(2048/16) = 128 = BLOCK_N exactly; workload_profile.md line 90-95 confirms all 6 T=7/8 are now 1-iter, and 4c46a94b is already 1-iter). `num_stages=2` software-pipelines a loop that has nothing to pipeline.

## Strategy

**Targeted fix.** Set `num_stages=1` on `_fused_split_combine_kernel` only (leave fused T≤2 kernel at `num_stages=2` — that one still iterates many times). One-line change at the launcher (line 374 of `solution/triton/sparse_fused.py`). No kernel edit. Exp_32's prior `num_stages=1` regression was at NS=8 where 2-iter loops existed and lost double-buffering; under NS=16 that regime does not exist (LESSON-54's "after structural rewrites, re-open tile axes that previously regressed").

## Actions (priority ordered)

### 1. `num_stages=1` on split+combine kernel (launcher line 374)

**What.** Edit `solution/triton/sparse_fused.py:374`:
```diff
-        num_warps=8, num_stages=2,
+        num_warps=8, num_stages=1,
```
The fused T≤2 kernel (line 337) stays at `num_stages=2` — that path iterates up to 32 times (BLOCK_N_FUSED=64 over 2048 TopK slots) and genuinely benefits from pipelining. This plan touches only the split+combine kernel.

**Why (mechanism).**
- **Split-loop iter count = 1 for all T≥3 under NS=16.** Per workload_profile.md lines 90-95, all 7 T≥3 stride-2 workloads have `per-split max valid ≤ 128 = BLOCK_N` → `max_bn = 128` → 1 iter. The dynamic `for bn in range(0, max_bn, BLOCK_N)` loop runs exactly once.
- **`num_stages=2` software-pipelines a dynamic range loop** by splitting one iter's work across two "stages" (tile load in stage i, MMA in stage i+1). With 1 actual iter, the pipeline prologue issues a speculative K load that never gets consumed, plus the epilogue must drain — both pure overhead.
- **Shmem budget freed.** `num_stages=2` allocates two K-tile buffers (2 × kc [128,512] bf16 ≈ 2×128 KB raw + overhead; Triton's async-copy scheduler may trim this, but the second stage definitely allocates some shmem). Freeing it allows higher SM occupancy or lets the compiler allocate more for the single-stage K load pipeline.
- **Combine loop is `tl.static_range(16)`** — unrolled at compile time, ignores `num_stages`. So this change is pure to the split loop.

**Why now (LESSON-54 applies).** LESSON-34 declared num_stages axis "fully closed" but cited exp_32 explicitly: exp_32 tested `num_stages=1` at NS=8 with stride-partition where T=8 tokens had 2-iter split loops (per-split valid up to 252 at NS=8). The prior regression was double-buffer-loss, specific to multi-iter loops. Under NS=16 that 2-iter regime is structurally eliminated (ceil(valid_max/16) ≤ 128 always, since TOPK=2048 caps valid_max). Re-opening this axis is justified by the structural change in NS.

**Impact estimate.** Profile says split_work = 9.33 µs with 2.25 µs compute gap over HBM floor. Some of that 2.25 µs is pipeline prologue/epilogue ops (stage init, load-drain). Freeing the second K-stage's shmem could also improve occupancy. Realistic range: **-0.3 to -1.0 µs per T≥3 workload** on split phase. On combine (uses static_range, unaffected), no change expected. Best case: 7 workloads × 0.7 µs ≈ 5 µs total saved. Worst case: neutral — pipelining's 1-iter overhead was already near-zero in Triton 3.6.

**Risk.** Two scenarios to watch:
- If ANY T≥3 workload in the full 128-set has per-split valid > 128, it runs 2-iter and regresses (same mechanism as exp_32). Profile checks: max valid_max in stride-2 is 2048 (02d6ae9c, 78b2e11c, 564007ac). Per-split at NS=16 = ceil(2048/16) = 128 → max_bn = 128 → 1 iter (boundary case). Boundary tokens should stay at 1 iter.
- If Triton 3.6 compiles `num_stages=1` with a slower default path for async-copy scheduling, could regress ~0.1-0.3 µs. Small compared to the pipelining savings.

## Do not try

- **num_stages=3, num_stages=4 on split kernel** — exp_19 tested =3, +5-6% regression via shmem over-allocation. Upward direction fully closed.
- **num_stages changes on fused T≤2 kernel** — exp_34 tested num_stages=3 on fused, +5-10% regression on T=1/T=2 (short-loop + shmem over-provisioning). Fused kernel stays at num_stages=2.
- **num_warps ≠ 8 on either kernel** — LESSON-26 strict optimum. Tested both directions (exp_14=4, exp_17=16).
- **Combine loop reformulations** — exp_52 (dynamic range), exp_53 (tl.reduce / offline softmax / two-pass). LESSON-55/56 fully closed this axis.
- **Adaptive NUM_SPLITS dispatch** — LESSON-57 closed via .item() catastrophe AND T-rule misdiagnosis.
- **BLOCK_N=64 on split** — exp_49 regressed +22%, LESSON-52 closed.
- **NUM_SPLITS=4, 32, or adaptive** — exp_31/51/54 closed the tile-size sweep.
- **PDL, cluster launch, Gluon, per-slot atomics** — LESSONS 27/29/41/45/46 block these architecturally.
- **cache_modifier='.cg' or evict_* variants on previously-tested sites** — LESSONS 47/48/49 close all four sub-axes.
- **Host-side valid_max inspection for dispatch** — LESSON-57 confirmed `.item()` adds ~90 µs via queue serialization.
- **D_CKV_SPLIT_FUSED=1** — exp_36 closed, +17-22% T≤2 regression.
- **`input_precision="ieee"` on bf16 dots** — LESSON-50 no-op for bf16×bf16.
- **Halving BLOCK_N_FUSED for low-valid T≤2** — LESSON-19 crossover at valid~100-500; current 64 is the chosen compromise.

## Coordination notes

- **Testing strategy: LESSON-51 stratification mandatory.** Stride-2 A/B × 2 runs same VM. Stratify by path (T≤2 fused = unchanged → noise baseline; T≥3 split-combine = affected).
  - Keep gate (T≥3 cluster, 7 workloads): mean Δ ≤ -0.0002 ms AND same-direction ≥5/7 in both runs AND no workload regresses >1.5% in either run.
  - Ignore T≤2 signals (unchanged kernel path); use them only as noise baseline per LESSON-51.
- **Revert gate.** Any T≥3 workload regressing >2% same-direction both runs, OR mean Δ ≥ 0 on T≥3 subset, OR correctness fail. Rollback is a single-line hunk.
- **If neutral (mean Δ within ±0.0001 ms).** Apply LESSON-50's rule: neutral + mechanism story (1-iter nothing-to-pipeline) is **not** a keep — revert and move on. This is not a sub-1% "direction-positive" keep like exp_43/48 because there's no multi-reader / cross-CTA mechanism making this a "durable" win.
- **Measurement protocol.**
  1. Quick first (2/2 correctness check; should pass — no numerical change).
  2. A/B stride-2 × 2 runs vs exp_51's snapshotted `experiments/exp_51/sparse_fused.py`.
  3. Only run full 23-workload benchmark if A/B gate passes.
- **Do NOT combine with any other change.** One optimization per iteration (LESSON-25). The `num_stages` change is intentionally isolated.
- **Correctness sanity.** abs_err should be byte-identical to exp_51 (1.56e-02). `num_stages` affects scheduling only, not math.

## Cited evidence

- **profile.md lines 12-28**: split_work = 9.33 µs, HBM K floor = 7.08 µs, gap = 2.25 µs. 2-iter claim on 05f6de65 is about the **fused path** (T=2, not T=8 as mislabeled) — does not conflict with the 1-iter-always claim for split-combine path.
- **workload_profile.md lines 83-95 (stride-2 iter-count table)**: all 7 T≥3 stride-2 workloads are 1-iter at NS=16 × BLOCK_N=128.
- **exp_32/result.md line 25**: exp_32 failed specifically because T=8 at NS=8 was **2-iter**. This regime no longer exists at NS=16.
- **LESSON-54**: "after structural rewrites (NS, partition scheme), re-open tile-size axes that previously regressed." Directly applicable: NS=16 (exp_51) is a structural rewrite relative to exp_32's NS=8 era.
- **LESSON-34**: declared num_stages closed; that declaration was specific to the pre-NS=16 regime where 2-iter loops existed. Under modern kernel state, that declaration is stale.
- **exp_51/result.md lines 33-36**: profile invalidation lesson — "when profile says 'at HBM floor,' ask 'whose bytes?'" Same principle: when LESSONS say "axis closed," ask "under which partition?" — structural rewrites change the axis's regime.
