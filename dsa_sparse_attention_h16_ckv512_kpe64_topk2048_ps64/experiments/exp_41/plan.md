# Plan — exp 41

## Diagnosis

Three consecutive reverts since exp_37 (exp_38 per-slot atomics, exp_39 partial_ml merge, exp_40 Gluon `bw.tcgen05_mma`) have closed the three biggest remaining targets under their assumed attack angles: the 1.8 µs barrier-spin bucket (atomic lever fully closed per LESSON-45; cluster-sync structurally blocked per LESSON-27 regardless of atomic presence — `TritonGPUPlanCTAPass` asserts on ANY tiled tensor op under `num_ctas>1`), the combine-IO axis (triple fail: exp_28/exp_30/exp_39), and the 5.51 µs split-MMA bucket (exp_40 LESSON-46 — HM=128 MMA padding + D-chunking + ~160 MMA-fences makes Gluon 6.1× slower at this H=16 shape). **The profile.md that named the 4.94 µs launch / 5.51 µs split / 2.23 µs barrier / 2.32 µs combine / 1.92 µs Q-load buckets is now 22 experiments stale.** Exp_26 (stride-partition) restructured the split-phase workload distribution; exp_37 (monotonic counter) shaved ~0.44 µs off the barrier bucket. The absolute bucket sizes have materially shifted, and with every scalar-knob axis closed, further wins require identifying buckets the old profile didn't name — which means fresh instrumentation is the prerequisite to any non-speculative next attack.

Two secondary observations support a diagnostic-first iteration rather than another speculative kernel edit: (1) the last three kernel-edit iterations have all been regressions despite well-defined hypotheses, indicating the optimizer is working from an outdated bucket picture; (2) the exp_40 post-mortem explicitly names three candidate angles (cluster-sync via inline PTX, warp-specialization, T-specific NUM_SPLITS) — all three are either structurally blocked (the first, per LESSON-27) or speculative on stale bucket data (the last two). **Re-profiling first gives the optimizer a non-speculative target for exp_42** rather than burning another iteration on a guess.

## Strategy

**Targeted fix: diagnostic re-profile + one minimal micro-ablation.** This iteration does two cheap things in sequence:

1. **Re-run `scripts/profile_kernel5.py` against the exp_37 baseline** to regenerate `experiments/profile.md` with fresh bucket attribution. No kernel edit for this step.
2. **Apply a one-line `cache_modifier=".cg"` to the Q_nope / Q_pe loads at lines 74–75 of `_fused_split_combine_kernel`.** This is the last explicitly-unexamined cache-modifier axis — exp_23 did `.cg` on K loads (6/7 large-T workloads directionally positive, below noise); exp_24 did `.cg` on partial_* stores (7/7 directionally positive, below noise). Q loads in the split kernel have never been tried. LESSON-40 supports `.cg` on SM-starved grids where lines are re-touched across CTAs (8 splits per token load the same Q); the L1-bypass saves a tag-check on data that's L2-resident anyway. Expected delta: **0 to −0.3%** (direction-positive or tie), same regime as exp_23/exp_24 — well within the sub-1% A/B confirmation band.

The real value of exp_41 is the refreshed profile. The `.cg` on Q is an almost-free piggyback probe that completes the cache-modifier sweep before we move on from it.

## Actions (priority ordered)

1. **Re-generate `experiments/profile.md` from exp_37 baseline.**
   - **What:** `modal run scripts/profile_kernel5.py` (no kernel edit). The harness builds stubbed variants of `_fused_split_combine_kernel` and derives per-phase µs by subtraction. Output: per-phase µs breakdown (launch, Q-load, split compute, barrier spin, combine compute) at T=1/T=2/T=6/T=8. Overwrite the stale `experiments/profile.md` with the new numbers.
   - **Why:** The current profile.md was taken at exp_15 baseline, pre-exp_26 (stride-partition) and pre-exp_37 (monotonic counter). The barrier spin bucket dropped ~0.44 µs per exp_37. Split compute distribution changed under stride-partition (max-CTA wall time reshaped). Without fresh bucket numbers, every next hypothesis ("the X µs bucket is now the biggest") is a guess. The four immediate prior regressions (exp_38/39/40) all worked from the stale profile's hypotheses.
   - **Impact:** Diagnostic. 2 Modal minutes. Produces the bucket map that informs exp_42. If the refreshed profile shows an unexpected bucket > 1 µs (e.g., softmax compute, partial-write HBM, LSE store), that becomes exp_42's target. If it confirms the old bucket map minus 0.44 µs, exp_42 becomes a structural re-think.

2. **Apply `cache_modifier=".cg"` to Q_nope and Q_pe loads in `_fused_split_combine_kernel` (split phase, lines 74–75 of `solution/triton/sparse_fused.py`).**
   - **What:** Two-line edit. Before: `q_nope = tl.load(q_nope_ptrs)` / `q_pe = tl.load(q_pe_ptrs)`. After: `q_nope = tl.load(q_nope_ptrs, cache_modifier=".cg")` / `q_pe = tl.load(q_pe_ptrs, cache_modifier=".cg")`. Applies only to the split-kernel Q loads; leaves `_fused_attn_kernel` (T≤2) Q loads unchanged per LESSON-40's regime-dependent rule.
   - **Why:** The Q_nope tile is 16 KB and Q_pe is 2 KB. 8 CTAs per token load the same Q. All 8 go through L1 (tag check per CTA) then hit L2 which is where the data actually lives. `.cg` bypasses L1 → direct L2. LESSON-40 established this helps on SM-starved grids where lines are re-touched across CTAs (exactly the split-kernel case); the effect on K loads in exp_23 was 6/7 large-T directionally positive at 0.02–0.18%. Q is an even cleaner fit — it's actually read by every split CTA of the same token (K rows are per-CTA different under stride-partition; Q is CTA-invariant). Q is loaded ONCE per CTA at prologue, kept in registers, so there's no intra-CTA L1 replay benefit to preserve — pure tag-check overhead to eliminate.
   - **Impact:** Expected direction-positive at 0.0–0.3% on T≥3 workloads. Never breakeven or measurable win a priori — same band as exp_23/exp_24. Small-T path untouched.

3. **A/B-confirm exp_41 vs exp_37 with `scripts/ab_benchmark.py --a experiments/exp_37/sparse_fused.py`.**
   - **What:** One paired stride-2 A/B run. Sub-1% deltas require paired measurement (CLAUDE.md rule); cross-VM single runs are noise.
   - **Why:** The expected delta is sub-1% based on prior cache-modifier experiments. Without paired A/B the signal is indistinguishable from VM noise.
   - **Impact:** ~2 Modal minutes. Confirms or rejects the `.cg`-on-Q hypothesis cleanly.

## Do not try

All prior closed axes reaffirmed (references to the exp/LESSON that closed them):

- **`NUM_SPLITS ≠ 8`** — exp_4 (+33-50%), exp_31 (−62% on T=8). LESSON-12.
- **`num_stages ≠ 2`** on either kernel — exp_19 (split=3), exp_32 (split=1), exp_34 (fused=3). All directions closed.
- **`num_warps ≠ 8`** on H=16 — exp_14 (=4), exp_17 (=16). LESSON-26.
- **`BLOCK_N=256`** on split — shmem OOM (exp_33).
- **`.cg` on combine-phase LOADS** — exp_28 regressed. LESSON-40.
- **Compact-block partition** (exp_27, exp_29) — full-TopK pre-scan exceeds coalescing savings.
- **Buffer persistence** (exp_30) — caching allocator masks cost.
- **`D_CKV_SPLIT_FUSED=1`** (exp_36) — D-parallelism in fused is load-bearing. LESSON-43.
- **Per-slot atomics in the same cache line** (exp_38) — L2 atomic unit serializes. LESSON-45.
- **Combine-IO axis** (exp_28, exp_30, exp_39 — triple fail) — fully closed.
- **`launch_pdl` / `griddepcontrol`** (exp_21, exp_25) — invisible to CUPTI. LESSON-41.
- **`num_ctas=N` cluster on current kernel** (exp_20) — `TritonGPUPlanCTAPass` asserts on tiled ops regardless of atomic presence. LESSON-27. Applies to `tl.inline_asm`-based cluster-sync too; the planner runs before the asm is emitted.
- **Merging partial_m/partial_l via `tl.join`/`tl.split`** (exp_39) — reshuffle, not work removal.
- **Gluon `bw.tcgen05_mma` as primary compute path** (exp_40) — HM=128 MMA padding + D-chunking uncompetitive at H=16. LESSON-46.
- **`torch.cat` of Q_nope + Q_pe on the host side to fuse the two loads** — adds host overhead > saved GPU µs. Classic anti-pattern.
- **Host-side dispatch on valid-count** — requires GPU-sync readback; costs more than any saving. (Reaffirmed from exp_39 plan.)
- **T=2 dispatch pivot to split+combine** — 7/8 T=2 workloads are low-valid (5–10 µs in fused); pushing them to split+combine would add barrier + combine overhead → regression. Only 1 outlier (05f6de65 @ 18.6 µs) would benefit, and the host can't detect it without reading indices.

## Revert threshold

- **Profile re-run has no revert threshold** — it's diagnostic; always keep the refreshed `profile.md`.
- **`.cg`-on-Q A/B:** keep if ≥ 6/12 B wins AND mean Δ ≤ 0 on same-VM paired run. Revert if either gate fails. Hard-revert at any workload showing > +1% regression (no second-run needed; that's clear signal).
- **If profile re-run shows the old bucket map unchanged (minus the known 0.44 µs barrier reduction):** accept exp_41 as a maintenance iteration, let exp_42 tackle the biggest named bucket.
- **If profile re-run reveals a new bucket > 1 µs that's not in the old map** (e.g., softmax overhead, LSE write contention, partial-write HBM at ≠ L2-resident volume): flag it, note it in result.md, make exp_42 target that bucket.

## Fallback if it fails

If the profile re-run shows no meaningful bucket shift AND the `.cg`-on-Q A/B ties (both likely based on prior precedent), the next experiment (exp_42) should target the **last structural lever the research agent has identified as genuinely unexplored: replace the atomic-barrier counter's current 4-byte int32 scalar with a 4-byte flag on a dedicated cache line per token (128-byte stride per slot, not the 4-byte stride exp_38 tried).** LESSON-45 closed 4-byte-stride per-slot atomics because they serialize on a single cache line. 128-byte stride per slot means 8 separate cache lines — 8 separate L2 atomic lanes in parallel. Cost: 8× counter tensor size (minor: kilobytes), 8× cache-line fetch per spin (bigger concern — 8 separate spin loads per iter). But LESSON-45 explicitly marked this as "plausibly worse overall — skipping this variant"; given all other axes are closed, it deserves one direct empirical test. Expected: ±0.5% on large T.

**If even that fails:** the kernel is effectively at its Triton ceiling for this workload distribution. The next research-agent synthesis should honestly recommend either (a) accept the current baseline as optimized for this workload distribution, or (b) request the user to expand the workload set so a broader distribution justifies a different kernel architecture. We've hit a **ceiling-is-lower-than-ambition** situation — the ambition is a 5–10% further win; the remaining Triton/Gluon levers give 0.5–1% at best. Do NOT pivot back to Gluon (closed per LESSON-46) or to cluster-sync (closed per LESSON-27). Do NOT re-sweep locked axes.

## Coordination notes

- **Quick iterations.** Action 1 is a profile-only Modal run (~2 min, no kernel edit). Action 2 is a two-line edit + quick correctness (~2 min) + stride-2 A/B (~2 min). Both completable in one iteration.
- **Profile re-run BEFORE the kernel edit.** The fresh profile.md data may reveal that the `.cg`-on-Q lever is tiny relative to a newly-visible bucket. If so, action 2 is still worth running (piggyback, cheap), but result.md should flag the next iteration's target based on the profile.
- **Do NOT bundle additional changes.** LESSON-27 (exp_20) and exp_38 showed how a "cheap secondary" change can confound the signal. One micro-change per iteration, strict.
- **Log both sub-results in exp_41's `result.md`.** Clearly separate "profile refresh" from ".cg-on-Q A/B." If the A/B ties, mark the cache-modifier axis fully closed; if it wins, kept as marginal improvement.
- **The optimizer should read `experiments/profile.md` AFTER the re-run** (before writing result.md) to pull the updated bucket numbers into the discovery section. This ensures subsequent research agents work from fresh data, not the stale exp_15 baseline.
- **If profile_kernel5.py fails to compile against the current (monotonic-counter) kernel shape** — e.g., its stubbed variants reference the old `atomic_add(-1)` decrement path — patch the harness first rather than reverting. Sub-agent delegation may be appropriate here since the harness edits are mechanical and orthogonal to the kernel.
