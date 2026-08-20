# Experiment 53 — 2026-04-17

**Description:** Attack combine's serial softmax-rescale critical path via three
reformulations: (A) `tl.reduce` tree-reduction with 3D broadcast, (B) offline
softmax decomposition (`tl.max` + scale + `tl.sum`), (C) two-pass loop with
pre-computed `m_global` then offline-scaled serial accumulate. Per `plan.md`.
Baseline: exp_51.

Motivated by profile.md's "2-stage combine tree" recommendation and LESSON-55's
"serial rescale chain is the bottleneck, not loop sugar." Hypothesis: tree or
offline patterns should parallelize the log2(16)=4-stage rescale chain, cutting
combine from 5.12 → ~3 µs.

## Results
- Pass: 2/2 for all three variants (quick)
- Mode: A/B stride 2 × 3 runs (one per variant)

### Regression summary (T≥3 cluster mean across 7 workloads)
| Variant | Approach | T≥3 Δ% | Cause |
|---|---|---|---|
| A | `tl.reduce` + broadcast m/l to 3D | +34-37% | Broadcasting makes per-step work 32× bigger for m/l; `tl.reduce` serializes internally |
| B | `tl.max` + scale + `tl.sum` | +10% | 32-KB `pacc_all * scale[:, :, None]` materialization defeats 2-KB rolling tile |
| C | Two-pass: pre-compute m_global, offline-scaled serial | +20-25% | Lost online-softmax ILP the compiler uses; still serial write-back to l_global/acc_comb |

All three reverted.

## Learnings

**Combine axis is now deeply closed** after 4 attempts (exp_4, exp_5, exp_52, exp_53).
Static_range + online-softmax is compiler-optimized:
- Online softmax pattern lets compiler schedule alpha/beta computation of iter N+1
  alongside acc update of iter N. 16-wide unroll gives deep ILP.
- 2-KB rolling tile residency matches B200's warp-register allocation tightly;
  any formulation that materializes 32-KB intermediate loses this.
- `tl.reduce` with a custom combine_fn does NOT guarantee tree parallelism — it
  appears to serialize into a fold at lower level on Triton 3.6.
- Offline softmax decomposition requires 32-KB temporaries that can't fit in the
  tight register budget used by the incremental online pattern.

**Profile.md's "2 µs from tree reduction" was optimistic.** The actual combine
compute ceiling is limited by the critical path of the softmax math, which the
compiler already optimizes close to the floor via static_range unroll.

**Real bottleneck insight:** If combine's 5.12 µs is compute-limited on the
critical path, the remaining lever is to **reduce the number of iters** (smaller
NUM_SPLITS) or **parallelize across warps** (intra-CTA structural rewrite).
Neither is a simple drop-in change.

## Decision
Revert to exp_51. Stop trying to rewrite the combine loop's scheduling pattern.

Next likely axis: structural combine parallelism at the warp level (each of 8
warps handles 2 of 16 partials into its own state, then cross-warp reduce via
shmem) — this is a bigger rewrite but gives REAL parallelism the compiler
can't extract from serial dependencies. OR: investigate split-phase optimization
(9.33 µs vs 7.08 µs HBM floor = 2.25 µs compute gap, up from 1.39 µs at NS=8).

LESSON-56 added.
