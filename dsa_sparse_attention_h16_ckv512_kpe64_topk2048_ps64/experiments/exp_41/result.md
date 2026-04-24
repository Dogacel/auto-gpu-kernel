# Experiment 41 — 2026-04-17

**Description:** Two-part diagnostic iteration per `plan.md`. (1) Re-ran `scripts/profile_kernel5.py` to overwrite stale exp_15-era `experiments/profile.md` with fresh bucket attribution against exp_37 baseline. (2) Applied `cache_modifier=".cg"` to Q_nope/Q_pe loads in `_fused_split_combine_kernel` (lines 74-75) — the last unexplored cache-modifier axis (after exp_23 K-loads, exp_24 partial stores, exp_28 combine-loads). Paired A/B vs exp_37.

## Results

### Part 1 — Profile refresh
- **Barrier cost dropped from 2.23 µs (stale) → 0.71 µs (fresh).** Exp_37's monotonic counter recovered 1.52 µs (68% of the bucket). Remaining 0.71 µs is wait-floor (slowest CTA sync), attackable only via cluster-sync (Gluon-only, blocked per LESSON-27).
- Split_work = 8.05 µs, combine_work = 4.69 µs, Q-load = 3.58 µs, launch floor = 6.35 µs. VM noise contributes uniform +1.4 µs across buckets vs stale, so **relative magnitudes unchanged**.
- **No new bucket revealed.** The refreshed profile confirms the exp_37 baseline is within ~1 µs of its Triton ceiling (14.5 µs CUPTI projected; current 15.5 µs observed).
- Profile.md overwritten at `experiments/profile.md`.

### Part 2 — .cg on Q loads
- Pass: 2/2 quick (abs_err identical to exp_37 within noise)
- Paired A/B stride-2 vs exp_37: **3/12 B wins, mean Δ = +0.0000 ms → A (exp_37) faster**

**A/B paired stride-2:**
| UUID | T-class | A (exp_37) | B (exp_41) | Δ | % |
|---|---|---|---|---|---|
| 02d6ae9c | T=8 | 0.0156 | 0.0155 | −0.0000 | **−0.17% B** |
| 05f6de65 | T=2 | 0.0187 | 0.0188 | +0.0000 | +0.24% |
| 0c23b10c | T=1 | 0.0053 | 0.0053 | +0.0000 | +0.36% |
| 2207f0fd | T=8 | 0.0155 | 0.0155 | +0.0000 | +0.10% |
| 232ed014 | T=8 | 0.0150 | 0.0151 | +0.0000 | +0.26% |
| 4c46a94b | T=6 | 0.0111 | 0.0111 | −0.0000 | **−0.11% B** |
| 5096e459 | T=8 | 0.0158 | 0.0158 | −0.0000 | **−0.14% B** |
| 564007ac | T=8 | 0.0159 | 0.0159 | +0.0000 | +0.06% |
| 78b2e11c | T=8 | 0.0155 | 0.0155 | +0.0000 | +0.04% |
| b7668cfd | T=1 | 0.0055 | 0.0055 | +0.0000 | +0.30% |
| e6b849f2 | T=2 | 0.0080 | 0.0080 | +0.0000 | +0.36% |
| f77df5ce | T=2 | 0.0054 | 0.0055 | +0.0000 | +0.83% |

All deltas within ±1% noise band. B wins only on 3 large-T workloads at sub-0.2% magnitudes. Revert gate failed (need ≥6/12 AND mean Δ ≤ 0); reverted.

### Mode: stride-2 A/B vs exp_37 + profile re-run

## Verdict

**Reverted to exp_37 baseline.** Confirmed byte-for-byte via `diff -q solution/triton/sparse_fused.py experiments/exp_37/sparse_fused.py`.

## Discoveries

1. **Cache-modifier axis fully closed.** Four tests: exp_23 `.cg` on K loads (tie, ~0.1% direction-positive), exp_24 `.cg` on partial stores (tie, ~0% direction-positive), exp_28 `.cg` on combine loads (+0.5% regression), exp_41 `.cg` on Q loads (tie, mean Δ=+0.0000 ms). The split-kernel Q loads benefit from L1 tag-check elimination on paper, but in practice the Q tile is small enough (16 KB + 2 KB) that L1 behavior doesn't measurably differ from L2-direct. Axis fully closed.

2. **Profile refresh: no new bucket revealed.** The main finding was expected — barrier bucket dropped 1.52 µs as predicted, other buckets stable with VM variance. This effectively confirms that the optimizer has been working from correct relative-magnitude intuition post-exp_37, and the missing quantitative update was the barrier number (which we can now cite as 0.71 µs rather than 2.23 µs in future plans).

3. **Triton ceiling estimate: ~14.5 µs CUPTI (projected) vs 15.5 µs observed.** Current kernel is within ~1 µs (6-7%) of its achievable Triton-only floor. Remaining 1 µs budget is distributed across 0.71 µs barrier wait-floor + ~0.3 µs slack in other buckets, none of which are individually attackable by known Triton axes.

4. **VM variance is material for profile interpretation.** The same-kernel noop floor drifted +1.41 µs (4.94 → 6.35) across VMs between exp_15 and exp_41. Any per-bucket comparison must account for this; use the RELATIVE sizes and only trust Δs within a single profile run.

## Next directions

- **Exp_42: try 128-byte-stride per-slot atomics (LESSON-45's explicitly-skipped variant).** Plan fallback: 8 separate cache lines → 8 separate L2 atomic lanes in parallel. Expected ±0.5% on large-T. This is a direct empirical probe of a clearly-marked unexplored micro-axis; even a neutral result fully closes the atomic lever.
- **Exp_43+ (if exp_42 ties/regresses):** the Triton axis is at its ceiling for this workload distribution. Remaining speculative moves are workload-specialization (T-specific NUM_SPLITS compile variants) or structural rewrites (fused T≤2 warp-split via `tl.inline_asm` with warp ID — brittle, untested). No more "obvious" plays.
- **Not retry:** any cache modifier anywhere (all 4 sub-axes tested). Any Gluon MMA-based compute path at H=16 (LESSON-46). Any `num_ctas`-based cluster (LESSON-27). Combine-IO axis.
