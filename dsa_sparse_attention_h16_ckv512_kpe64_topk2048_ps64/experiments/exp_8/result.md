# Experiment 8 — 2026-04-16

**Description:** D_CKV_SPLIT tuning. exp_7 chose D_CKV_SPLIT=4 (BLOCK_D=128) heuristically. This experiment sweeps to 8 (BLOCK_D=64) and 16 (BLOCK_D=32) to find the sweet spot for combine-kernel parallelism. 

Keeps all other parameters from exp_7 constant.

## Results (D_CKV_SPLIT=8 — chosen as new best)
- Pass: 12/12
- Kernel latency (ms): small=0.009 / large=0.020 / overall=0.009 (min) / 0.020 (median) / 0.020 (max)
- Reference latency (ms): not profiled
- Max abs err: 1.56e-02
- Mode: stride 2 (12 workloads); A/B confirmed on same VM

**A/B D_CKV_SPLIT=8 (B) vs D_CKV_SPLIT=4 (A, exp_7):**
| UUID | A (D=4) | B (D=8) | Δ |
|---|---|---|---|
| 0c23b10c (T=1) | 0.0160 | 0.0118 | **−26.70%** ✅ |
| 02d6ae9c (T≥6) | 0.0232 | 0.0216 | −7.08% ✅ |
| 232ed014 (T≥6) | 0.0234 | 0.0214 | −8.58% ✅ |
| 4c46a94b (T=6) | 0.0234 | 0.0213 | −8.60% ✅ |
| 5096e459 (T≥6) | 0.0233 | 0.0215 | −7.64% ✅ |
| 564007ac (T≥6) | 0.0240 | 0.0217 | −9.78% ✅ |
| b7668cfd (T=2) | 0.0132 | 0.0124 | −6.04% ✅ |
| e6b849f2 (T=2) | 0.0168 | 0.0150 | −10.99% ✅ |
| f77df5ce (T=2) | 0.0136 | 0.0164 | +20.63% ❌ (noise? stride2 showed 0.010 for same uuid) |
| 78b2e11c (T≥6) | 0.0240 | 0.0232 | −3.06% ✅ |
| 2207f0fd (T≥6) | 0.0237 | 0.0241 | +1.64% ≈ |
| 05f6de65 (T=2) | 0.0232 | 0.0224 | −3.49% ✅ |

**Summary:** B wins 10/12, mean Δ = −0.0013 ms (−6%). The f77df5ce +21% in A/B appears to be variance (stride-2 reported 0.010 for the same workload at D_SPLIT=8).

**A/B D_CKV_SPLIT=16 (B) vs D_CKV_SPLIT=4 (A, exp_7):**
| Group | Mean Δ |
|---|---|
| Large T=6-8 (8 workloads) | −0.6 to −2.5% (tiny wins) |
| Small T=1-2 (4 workloads) | +1.4 to +4.0% (regression) |
| Overall | mean −0.5%, 8/12 wins |

So D_CKV_SPLIT=16 is worse than 8 on small workloads (each CTA too tiny at 16×32=512 elems) and similar on large. 8 is the sweet spot.

## Learnings
- **D_CKV_SPLIT=8 (BLOCK_D=64) is the sweet spot.** At BLOCK_D=64, each combine CTA has 16×64=1024 fp32 acc to update per iter — enough to keep warps busy, small enough to run 8 CTAs in parallel per token on 148 SMs.
- **BLOCK_D=32 is too small.** The per-CTA work (16×32=512 fp32) doesn't saturate even a single warp-group; per-CTA prologue (pointer arithmetic, register setup) relatively dominates.
- **BLOCK_D=128 had too few CTAs** (4/token) — not enough parallelism vs the available SM headroom.

## New best: 0.009 ms (smallest) / 0.020 ms (median) / 0.020 ms (max).

Cumulative speedup path:
- exp_1 (fused Triton): 0.100 ms
- exp_2 (split-K=8): 0.024 ms (−76%)
- exp_6 (dynamic loop bound): 0.022 ms (−8% vs exp_2)
- exp_7 (D-parallel combine, D=4): 0.021 ms (−5% vs exp_6)
- exp_8 (D-parallel combine, D=8): 0.020 ms (−5% vs exp_7)

## Next candidate axes (exp_9+)
1. **Combine num_warps = 2** — per-CTA work has shrunk to H×BLOCK_D=1024 ops, num_warps=4 may have idle warps. Try 2.
2. **Split kernel prologue optimization** — Q-tile is loaded 8× redundantly per token. Use cluster_dims(NUM_SPLITS) to share Q-tile via DSMEM? Or reduce Q-tile load latency.
3. **Skip partial_acc store for num_valid=0 splits** — saves 32 KB × N_empty_splits HBM writes per token. Adds a conditional epilogue to split kernel.
4. **Tune num_stages in combine now that body is smaller** — try 1 vs 2.
5. **Re-profile after combine-phase relief** — split kernel is now the dominant phase (~12-20 µs). Profile may show a new bottleneck.
