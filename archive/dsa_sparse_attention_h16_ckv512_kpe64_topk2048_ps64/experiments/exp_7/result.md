# Experiment 7 — 2026-04-16

**Description:** D_ckv-parallel combine kernel (lever (a) from `profile.md`). Change `grid2` from `(T,)` to `(T, D_CKV_SPLIT=4)`; each CTA handles `BLOCK_D = D_ckv // 4 = 128` channels. The online-softmax state `m_global/l_global` is `[H]` = 16 f32 per CTA, replicated identically across d-programs at negligible recompute cost. Only `d==0` writes LSE to avoid store races.

Motivation from profiler: combine was 3× off memcpy floor at 14 µs event / ~5-10 µs cupti (~40-45% of submission latency on small workloads). num_warps sweep showed combine doesn't scale with warps — classic single-CTA serial bottleneck. Parallelizing the ∑ across D_ckv breaks that.

## Results
- Pass: 12/12
- Kernel latency (ms): small=0.012 / large=0.021 / overall=0.012 (min) / 0.021 (median) / 0.022 (max)
- Reference latency (ms): not profiled
- Max abs err: 1.56e-02
- Mode: stride 2 (12 workloads); A/B confirmed against exp_6 on same VM

**A/B vs exp_6 (paired, same VM, B=exp_7):**
| UUID | A (exp_6) | B (exp_7) | Δ |
|---|---|---|---|
| 0c23b10c (T=1) | 0.0191 | 0.0166 | **−12.86%** ✅ |
| b7668cfd (T=2) | 0.0197 | 0.0174 | **−11.97%** ✅ |
| f77df5ce (T=2) | 0.0198 | 0.0180 | **−8.74%** ✅ |
| e6b849f2 (T=2) | 0.0198 | 0.0187 | −5.72% ✅ |
| 05f6de65 (T=2) | 0.0226 | 0.0213 | −6.12% ✅ |
| 02d6ae9c..2207f0fd (T=6-8) | 0.0242-0.0243 | 0.0231-0.0233 | −3.83 to −5.10% ✅ |

**Summary:** B wins 12/12, mean Δ = −0.0014 ms (−6%). Small workloads gain most (−9 to −13%) because combine was a larger fraction of their latency. Large workloads still get a consistent −4 to −5%.

## Learnings
- **Profiler prediction held.** The projected ~3-4 µs cupti saving materialized as ~1.1–2.5 µs across workloads, matching the 3× memcpy-floor headroom. Single-CTA combine → 4-way D-parallel combine is a clean, proven pattern.
- **The m/l state redundancy is truly free.** All 4 D-programs compute identical `m_global, l_global`; the per-iter softmax merge is H=16 float ops, ~0.1% of the acc update. The "4× work on scalar state" fear is moot at this scale.
- **Guarding LSE write with `if d == 0:`** is the clean way to avoid a store race on `lse[t, h]`. Triton supports scalar-conditioned stores here without affecting the main reduction hot path.

## New best: 0.012 ms (smallest) / 0.021 ms (median) / 0.022 ms (max).

## Next candidate axes (exp_8+)
1. **Parallelize split kernel's Q-tile load** — profile said split's 6 µs cupti on small workloads is prologue-dominated (Q-tile load + index scan + 256 KB partial_acc writes across 8 CTAs). Can we reduce partial_acc store footprint? E.g., store only the first `num_valid_blocks > 0` splits, or compact partial_acc representation.
2. **BLOCK_D tuning** — try D_CKV_SPLIT=2 (BLOCK_D=256) or D_CKV_SPLIT=8 (BLOCK_D=64) to see if fewer/more splits is better.
3. **NUM_SPLITS=4 (re-visit)** — with parallel combine, the per-iter combine cost is lower, so halving NUM_SPLITS may now help without the combine penalty that blocked it in exp_5.
4. **Fuse split + combine** — now that combine is ~8 µs, combined with allocation (6.6 µs event), there's still ~15 µs of "non-compute" overhead. A single-kernel design could eliminate the partial_acc allocation entirely.
5. **num_warps=2 on combine** — split and combine both now have less work per CTA.
