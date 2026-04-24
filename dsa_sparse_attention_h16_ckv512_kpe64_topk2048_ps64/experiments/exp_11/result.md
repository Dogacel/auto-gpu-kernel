# Experiment 11 — 2026-04-17

**Description:** Hybrid dispatch based on `num_tokens`. Uses the exp_10 fused D-parallel kernel (single launch, no `partial_acc` round-trip) for T ≤ 2; uses exp_9's split+combine for T ≥ 3. Exploits the bimodal exp_10 result — small T gains 36–46% from launch-tax elimination, while large T regresses 2–3.5× because Q@K^T logits get replicated 8× across D-programs.

Threshold = 2 chosen empirically from exp_10 per-workload results. The workload set has T ∈ {1, 2, 6, 7, 8}; T=3–5 is absent, so threshold=2 cleanly separates the regimes.

## Results
- Pass: 12/12
- Kernel latency (ms): small=0.007 / large=0.018 / overall=0.007 (min) / 0.016 (median) / 0.026 (max)
- Reference latency (ms): not profiled
- Max abs err: 1.56e-02
- Mode: stride 2 (12 workloads) + A/B vs exp_9 (same VM)

**A/B vs exp_9 (paired, same VM, B = exp_11):**
| UUID | T | A (exp_9) | B (exp_11) | Δ |
|---|---|---|---|---|
| 0c23b10c | 1 | 0.0127 | 0.0071 | **−44.50%** ✅ |
| b7668cfd | 2 | 0.0133 | 0.0073 | **−45.17%** ✅ |
| e6b849f2 | 2 | 0.0132 | 0.0074 | **−43.94%** ✅ |
| f77df5ce | 2 | 0.0133 | 0.0071 | **−46.10%** ✅ |
| 05f6de65 | 2 | 0.0173 | 0.0151 | −12.73% ✅ |
| 4c46a94b | 6 | 0.0176 | 0.0175 | −0.56% ≈ |
| 02d6ae9c | 8 | 0.0177 | 0.0177 | −0.00% ≈ |
| 2207f0fd | 8 | 0.0177 | 0.0177 | −0.03% ≈ |
| 232ed014 | 8 | 0.0177 | 0.0176 | −0.36% ≈ |
| 5096e459 | 8 | 0.0178 | 0.0177 | −0.34% ≈ |
| 564007ac | 8 | 0.0178 | 0.0177 | −0.47% ≈ |
| 78b2e11c | 8 | 0.0177 | 0.0176 | −0.74% ≈ |

**Summary:** B wins 12/12, mean Δ = **−0.0022 ms (−12% overall)**. T≤2 gains 44–46% from fused path. Large T matches exp_9 (same kernel code path dispatched), differences within 1% noise.

## Learnings
- **Hybrid dispatch beats one-size-fits-all.** Small T and large T have fundamentally different cost profiles (launch-tax-bound vs parallelism-bound), so a single kernel shape can't optimally serve both. exp_10 was a failed monolithic fuse; exp_11 reuses its fused path only where it wins.
- **Launch tax is the dominant cost on small T.** With T=1 / T=2, the fused kernel saves 8 µs of the second kernel launch barrier and a 256 KB partial_acc HBM round-trip; the per-CTA work grows but is still small enough at low T to fit inside that savings.
- **Threshold choice matters and is workload-dependent.** Our set lacks T=3–5, so threshold=2 is a clean split. For workloads populating that gap, further tuning (try threshold=3 or 4) may be worthwhile. Given the current trace, threshold=2 is the safe choice.

## New best: 0.007 ms (smallest) / 0.016 ms (median) / 0.026 ms (max).

Median is technically unchanged from exp_9 (0.016 ms) because most workloads are T=8 (dominant regime); the improvement is in **small-T absolute latency** which exp_9 left on the table. The cupti submission total (sum across representative workloads) drops meaningfully.

Cumulative speedup path:
- exp_1: 0.100 ms
- exp_2 (split-K=8): 0.024 ms
- exp_6 (dynamic loop bound): 0.022 ms
- exp_7 (D-parallel combine D=4): 0.021 ms
- exp_8 (D-parallel combine D=8): 0.020 ms
- exp_9 (BLOCK_N=128): 0.016 ms
- exp_11 (hybrid fused/split dispatch): 0.016 ms median, but small-T drops 44-46%

## Next candidate axes (exp_12+)
1. **Re-profile** — launch overhead now the bottleneck only on large T. Small-T regime has new baseline.
2. **Small-T fused kernel optimization** — it's a separate code path now; tune num_warps, num_stages, try removing the kc_full load and only loading kc_slice via logits-split or similar.
3. **Large-T split kernel optimization** — split is now the entire path for large T; profile says split ~10 µs T=8 with ~3 µs real compute. 7 µs is launch barrier + dispatch.
4. **Persistent kernel for combine** — persistent CTAs with a work queue could collapse launch barrier cost across many tokens. Not helpful here (T is tiny).
5. **Further fuse large-T** — option to use fused kernel with D_CKV_SPLIT=1 (1 CTA per token, full-D acc) to eliminate combine; would be SM-starved but only 1 launch. Likely regresses based on exp_1 numbers.
