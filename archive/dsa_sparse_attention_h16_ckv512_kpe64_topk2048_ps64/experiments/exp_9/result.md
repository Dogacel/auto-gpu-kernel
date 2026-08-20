# Experiment 9 — 2026-04-16

**Description:** Double BLOCK_N from 64 to 128. Halves the iter count per split for tokens with many valid entries. Cost: more per-iter compute (two dots scale linearly with N) and more wasted compute on tokens with few valid entries (last block has more padding).

Keeps all other parameters from exp_8 constant (NUM_SPLITS=8, D_CKV_SPLIT=8, num_warps=8 split / 4 combine, num_stages=2 split / 1 combine).

## Results
- Pass: 12/12
- Kernel latency (ms): small=0.011 / large=0.017 / overall=0.011 (min) / 0.016 (median) / 0.017 (max)
- Reference latency (ms): not profiled
- Max abs err: 1.56e-02
- Mode: stride 2 (12 workloads); A/B confirmed against exp_8 on same VM

**A/B vs exp_8 (paired, same VM, B=exp_9):**
| UUID | A (BLOCK_N=64) | B (BLOCK_N=128) | Δ |
|---|---|---|---|
| 4c46a94b (T=6) | 0.0271 | 0.0176 | **−35.05%** ✅ |
| 02d6ae9c (T≥6) | 0.0217 | 0.0177 | −18.61% ✅ |
| 2207f0fd (T≥6) | 0.0217 | 0.0177 | −18.57% ✅ |
| 232ed014 (T≥6) | 0.0217 | 0.0176 | −18.75% ✅ |
| 564007ac (T≥6) | 0.0217 | 0.0179 | −17.77% ✅ |
| 78b2e11c (T≥6) | 0.0217 | 0.0176 | −18.77% ✅ |
| 5096e459 (T≥6) | 0.0217 | 0.0177 | −18.29% ✅ |
| 05f6de65 (T=2) | 0.0214 | 0.0174 | −18.59% ✅ |
| e6b849f2 (T=2) | 0.0151 | 0.0134 | −11.34% ✅ |
| 0c23b10c (T=1) | 0.0109 | 0.0129 | **+18.50%** ❌ |
| b7668cfd (T=2) | 0.0114 | 0.0134 | +17.72% ❌ |
| f77df5ce (T=2) | 0.0118 | 0.0133 | +13.05% ❌ |

**Summary:** B wins 9/12, mean Δ = **−0.0028 ms (−15%)**. Large workloads (T≥6) all win 17-35%. Small T=1 and 2 of 4 T=2 workloads regress 13-18% (tokens with too few valid entries waste compute on padding). Net median 0.016 ms vs 0.020 ms previous.

## Learnings
- **BLOCK_N=128 is a big win on workloads with many valid entries per split.** Fewer iterations → less per-iter prologue overhead, better tensor-core utilization (bigger K dimension per dot).
- **Cost: wasted compute on small workloads.** For a split with 33 valid entries, BLOCK_N=128 computes 128 positions (95 wasted) instead of 64 (31 wasted). Compute scales ~2× per block for a ~15% actual slowdown (overhead still dominates).
- **Adaptive BLOCK_N would be ideal but Triton requires constexpr.** Could launch two separately-compiled kernel variants and dispatch based on num_valid, but that's a complex specialization. The net-positive (-15% mean) suggests accepting the small-workload penalty is the right call for overall median.

## New best: 0.011 ms (smallest) / 0.016 ms (median) / 0.017 ms (max).

Cumulative speedup path:
- exp_1 (fused Triton): 0.100 ms
- exp_2 (split-K=8): 0.024 ms (−76%)
- exp_6 (dynamic loop bound): 0.022 ms
- exp_7 (D-parallel combine D=4): 0.021 ms
- exp_8 (D-parallel combine D=8): 0.020 ms
- exp_9 (BLOCK_N=128): 0.016 ms (−20% vs exp_8)

## Next candidate axes (exp_10+)
1. **Specialize BLOCK_N per num_valid**: compile two kernel variants and dispatch. Would reclaim the 13-18% lost on T=1/2 without giving up large wins.
2. **Re-profile** — phase breakdown may have shifted again.
3. **NUM_SPLITS=4** with new BLOCK_N=128. Each split is 512/128=4 blocks, which may be cleaner pipeline-wise and reduces combine fan-in.
4. **Small-workload fused kernel** (single kernel for T≤2, skipping split+combine split).
5. **num_warps tuning** — at larger BLOCK_N, per-CTA compute is larger, may benefit from more warps (16 on split?).
