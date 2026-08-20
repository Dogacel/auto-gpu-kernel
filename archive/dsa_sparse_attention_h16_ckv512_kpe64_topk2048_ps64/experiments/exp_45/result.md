# Experiment 45 — 2026-04-17

**Description:** Per `exp_44/result.md` next direction. Applied `eviction_policy="evict_first"` to per-iter `idx` load (line 92) ONLY, keeping the prefix scan (line 86) un-hinted. Hypothesis: per-iter load IS the last read of each idx row (loop consumes scan-produced data 128 elts at a time, KV gather fires immediately after; idx not re-read in combine), so hint should be safe. Keeping scan un-hinted avoids the L2-eviction-before-reuse pattern from exp_44.

## Results
- Pass: 2/2 quick (abs_err identical)
- Mode: stride-2 A/B vs exp_43
- **A/B 1/12 B wins, mean Δ = +0.0001 ms → A (exp_43) faster**

**A/B paired stride-2 vs exp_43:**
| UUID | T-class | A (exp_43) | B (exp_45) | Δ | % |
|---|---|---|---|---|---|
| 02d6ae9c | T=8 | 0.0153 | 0.0154 | +0.0001 | +0.35% |
| 05f6de65 | T=2 | 0.0185 | 0.0186 | +0.0000 | +0.24% |
| 0c23b10c | T=1 | 0.0055 | 0.0055 | +0.0000 | +0.83% |
| 2207f0fd | T=7 | 0.0154 | 0.0156 | +0.0002 | +1.00% |
| 232ed014 | T=8 | 0.0150 | 0.0151 | +0.0001 | +0.70% |
| 4c46a94b | T=6 | 0.0108 | 0.0109 | +0.0001 | +0.62% |
| 5096e459 | T=8 | 0.0156 | 0.0158 | +0.0002 | +1.09% |
| 564007ac | T=8 | 0.0157 | 0.0158 | +0.0001 | +0.92% |
| 78b2e11c | T=8 | 0.0153 | 0.0155 | +0.0002 | +1.13% |
| b7668cfd | T=2 | 0.0055 | 0.0055 | +0.0000 | +0.00% |
| e6b849f2 | T=2 | 0.0080 | 0.0079 | −0.0001 | **−0.96% B** |
| f77df5ce | T=2 | 0.0054 | 0.0054 | +0.0000 | +0.35% |

T≥3 (combine-kernel workloads, 6 of 7): consistent +0.6–1.1% regression. T≤2 (fused kernel, unaffected): ±1% noise. Smaller magnitude than exp_44's +3.5% (which hit both scan and per-iter), but still a clean directional loss on the touched path.

## Verdict

**Reverted to exp_43 baseline.** `solution/triton/sparse_fused.py` restored via `cp experiments/exp_43/sparse_fused.py solution/triton/sparse_fused.py`. Byte-for-byte verified.

## Discoveries

1. **Per-iter `idx` load is too small (512 B = 128 × int32) for L2 eviction hints to pay back.** Mechanism: L2 line is 128 bytes; 128 int32s = 4 lines. Freeing 4 lines (512 B) after the load doesn't relieve any measurable pressure — at this scale, the hint's benefit (HBM bandwidth freed for K gather) is smaller than its overhead (instruction-level routing, possibly disrupting prefetcher-driven line retention patterns). Contrast with K-load in exp_43: 144 KB/iter per CTA × 8 CTAs = ~1.1 MB of L2 freed per iter, materially useful.

2. **`evict_first` has a non-zero per-load cost.** At the scale where the hint is theoretically neutral (load is one-shot, frees little), it registers as a small regression, implying the hint mechanism itself costs instruction decode bandwidth / adds a metadata bit to each memory request that slightly slows the data-path. Consistent across 6 of 6 combine workloads at ~1%.

3. **Rule refinement for LESSON-48.** `evict_first` is only profitable when (a) load is truly one-shot within the kernel AND (b) freed L2 capacity is meaningful (tens of KB+). Both conditions must hold. exp_43 satisfies both. exp_44 fails (a). exp_45 satisfies (a) but fails (b). The "is it one-shot" heuristic is necessary but not sufficient — size matters too.

## Next directions

- **Exp_46: `evict_last` on partial_m/partial_l STORES (lines 124-125).** Multi-consumer pattern (1 writer × 8 combine readers across atomic barrier). Store hint tells L2 to keep these lines resident; combine CTAs read them immediately after barrier. Could reduce combine-phase L2 miss rate. These stores are small per-CTA (`partial_m[s]: [H]` fp32 = 64 B; `partial_l[s]: [H]` fp32 = 64 B) but total across splits is 1 KB × 8 CTAs = 8 KB per token — sized such that L2 retention is plausible.
- **Exp_47: `input_precision="ieee"` on the 3 `tl.dot` calls** (lines 101, 102, 112). Genuinely untested precision knob, fallback from exp_43's original plan.
- **Exp_48: `evict_last` on Q_nope/Q_pe loads (lines 74-75).** Q is loaded once per (t, s, d), held in registers through split phase, not re-fetched. But same Q rows are loaded by 8 D-CTAs simultaneously (same `t`, different `s`, same `d` range) → cross-CTA L2 sharing opportunity. Hint keeps lines pinned for the 8-way fan-out.
- **Not retry:** any eviction_policy on small (<2 KB) one-shot loads (per exp_45 discovery #2).
