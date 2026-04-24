# Experiment 44 — 2026-04-17

**Description:** Per `exp_43/result.md` next direction. Extended `eviction_policy="evict_first"` to both index loads in `_fused_split_combine_kernel`: `idx_scan` (line 86, full SPLIT_SIZE=256 prefix scan) and per-iter `idx` (line 92, BLOCK_N=128 per loop iter). Hypothesis: indices are one-shot from kernel's perspective, same mechanism as exp_43's K-load win. Exp_44 kept exp_43's `evict_first` on kc/kp and added it to both idx loads.

## Results
- Pass: 2/2 quick (abs_err identical)
- Mode: stride-2 A/B vs exp_43
- **A/B 3/12 B wins, mean Δ = +0.0003 ms → A (exp_43) faster**

**A/B paired stride-2 vs exp_43:**
| UUID | T-class | A (exp_43) | B (exp_44) | Δ | % |
|---|---|---|---|---|---|
| 02d6ae9c | T=8 | 0.0155 | 0.0160 | +0.0005 | +3.33% |
| 05f6de65 | T=2 | 0.0186 | 0.0186 | −0.0000 | **−0.12% B** |
| 0c23b10c | T=1 | 0.0052 | 0.0052 | −0.0000 | **−0.74% B** |
| 2207f0fd | T=7 | 0.0154 | 0.0160 | +0.0006 | +3.75% |
| 232ed014 | T=8 | 0.0150 | 0.0155 | +0.0005 | +3.26% |
| 4c46a94b | T=6 | 0.0110 | 0.0114 | +0.0004 | +3.41% |
| 5096e459 | T=8 | 0.0156 | 0.0162 | +0.0006 | +3.59% |
| 564007ac | T=8 | 0.0157 | 0.0163 | +0.0005 | +3.48% |
| 78b2e11c | T=8 | 0.0153 | 0.0159 | +0.0006 | +3.76% |
| b7668cfd | T=2 | 0.0054 | 0.0054 | +0.0000 | +0.30% |
| e6b849f2 | T=2 | 0.0079 | 0.0080 | +0.0000 | +0.16% |
| f77df5ce | T=2 | 0.0053 | 0.0053 | −0.0000 | **−0.18% B** |

Consistent +3.3–3.8% regression on 6 of 7 T≥3 workloads (4c46a94b the exception at +3.41% — still a regression). T≤2 workloads (fused kernel unchanged) show expected ±0.7% cross-VM noise.

## Verdict

**Reverted to exp_43 baseline.** Confirmed byte-for-byte via `diff -q solution/triton/sparse_fused.py experiments/exp_43/sparse_fused.py`.

## Discoveries

1. **`idx_scan` and per-iter `idx` read the SAME underlying data — `evict_first` on scan causes L2 miss on per-iter refetch.** Under stride-partition: scan reads `offs_split = s + arange(SPLIT_SIZE) * NUM_SPLITS = s + arange(256) * 8` (256 int32s). Per-iter reads `s + (bn + arange(128)) * 8` for bn ∈ {0, 128}. These are exactly the same data — the 256-element scan IS what the loop later consumes 128 at a time. With `evict_first` on scan, the L2 lines get pushed down the LRU; by the time the per-iter load fires (after 1 iter of K/V gather creates 144 KB pressure), the lines have evicted, forcing an HBM refetch. HBM latency ~500 ns × 8 CTAs × 2 iters = ~8 µs accumulated per call — matches observed +3.5% (≈ +0.0005 ms).

2. **`cache_modifier=".cg"` has no `evict_first`-equivalent semantics.** `.cg` on the scan would ALSO bypass L1, but the L2 line population is the same. So the regression is specifically the L2 hint interacting badly with the re-read pattern. Mechanism differs from LESSON-48.

3. **Eviction-policy hint should only be applied to TRULY one-shot loads.** K gather under stride-partition is one-shot (disjoint rows across iters, never re-read). Index scan is NOT one-shot (re-read by the loop). Applying evict_first indiscriminately regresses. Rule: `grep -A2 "tl.load" | grep -c` the data → how many reads hit this tensor over the kernel's lifetime? Apply evict_first only at 1-read count.

## Next directions

- **Exp_45: `evict_first` on per-iter `idx` ONLY (not the scan).** The per-iter load IS the last read of each idx row (KV gather happens immediately after; the idx tensor isn't touched again in the combine phase). Scan must stay un-hinted so the per-iter loads hit L2. Cleaner test of the axis; should be neutral-to-positive.
- **Exp_46 (if exp_45 ties): `evict_last` on Q_nope/Q_pe** (lines 74-75) — dual hint for multi-use data. Q is loaded once, held in registers for split phase; the L2 line isn't re-touched but Q is loaded by 8 D-CTAs simultaneously so `evict_last` could help cross-CTA L2 sharing.
- **Exp_47 (if exp_46 ties): `input_precision="ieee"` on the 3 `tl.dot` calls** (lines 101/102/112) — per exp_43 plan fallback; genuinely untested precision knob.
- **Not retry:** applying eviction_policy to any load that's re-read within the kernel's own lifetime.
