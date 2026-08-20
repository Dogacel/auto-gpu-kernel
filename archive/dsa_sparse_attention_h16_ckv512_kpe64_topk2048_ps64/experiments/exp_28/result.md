# Experiment 28 — 2026-04-17

**Description:** Add `cache_modifier=".cg"` to the three `tl.load(partial_*)` calls in the combine phase of `_fused_split_combine_kernel` (lines 155-156, 165). Complements exp_24 (`.cg` on split-phase stores) — split writes partials via `.cg` (L2, bypass L1), combine then reads with default `.ca` (L1+L2) which wastes an L1 lookup that can never hit. Setting `.cg` on the load skips the doomed L1 probe and goes directly to L2.

## Results
- Pass: 2/2 quick (0c23b10c, 2207f0fd)
- Mode: quick + A/B vs exp_26 (stride-2)
- **Reverted** — marginal regression

**A/B run vs exp_26 (stride-2):**
| UUID | A (exp_26) | B (exp_28) | Δ | % |
|---|---|---|---|---|
| 02d6ae9c | 0.0159 | 0.0160 | +0.0001 | +0.73% |
| 05f6de65 | 0.0188 | 0.0188 | −0.0000 | −0.10% |
| 0c23b10c | 0.0055 | 0.0056 | +0.0000 | +0.58% |
| 2207f0fd | 0.0159 | 0.0161 | +0.0002 | +1.18% |
| 232ed014 | 0.0155 | 0.0157 | +0.0002 | +1.55% |
| 4c46a94b | 0.0113 | 0.0117 | +0.0004 | +3.22% |
| 5096e459 | 0.0161 | 0.0164 | +0.0002 | +1.27% |
| 564007ac | 0.0162 | 0.0164 | +0.0002 | +1.07% |
| 78b2e11c | 0.0159 | 0.0160 | +0.0001 | +0.90% |
| b7668cfd | 0.0055 | 0.0055 | +0.0000 | +0.29% |
| e6b849f2 | 0.0081 | 0.0080 | −0.0001 | −1.15% |
| f77df5ce | 0.0054 | 0.0055 | +0.0000 | +0.89% |

Paired: A wins 10/12, mean Δ = +0.0001 ms (B slower). Consistent direction.

## Design (reverted)

```diff
     for si in tl.static_range(NUM_SPLITS):
-        m_si = tl.load(pm_ptr_s)
-        l_si = tl.load(pl_ptr_s)
+        m_si = tl.load(pm_ptr_s, cache_modifier=".cg")
+        l_si = tl.load(pl_ptr_s, cache_modifier=".cg")
         ...
-        acc_si = tl.load(pacc_ptr_s)
+        acc_si = tl.load(pacc_ptr_s, cache_modifier=".cg")
```

## Discoveries

1. **`.cg` on combine-phase LOADS marginally regresses** (~1% on average, +3.22% on the stride-win workload 4c46a94b). Direction opposite of expectation.

2. **Plausible explanations for regression**:
   - The `.cg` load emits `ld.global.cg` PTX with slightly different latency/throughput than `ld.global.ca`. On SM_100a (B200), `.ca` may still check L1 tags but fall back cheaply to L2 if the split write's line isn't present — and L1 tag check is not on the critical path if MMA/softmax overlaps it. `.cg` forces explicit L2-only semantics which might serialise slightly.
   - Alternative: PCIe-class MMIO-style loads; or the Triton 3.6 compiler emits a different scheduling hint for `.cg` that disables some latency hiding.

3. **Important asymmetry with exp_24**: On `tl.store`, `.cg` is clearly correct (write-through semantics, no L1 pollution from reads you won't revisit). On `tl.load`, the default `.ca` is already tuned for loads that might reuse — `.cg` is only strictly faster when you *know* L1 will miss AND the tag check is on the critical path, neither of which is true here (combine reads each partial exactly once and tag check overlaps with ongoing MMA/softmax).

4. **Cache-modifier axis is saturated.** Split-phase K loads (.cg) + split-phase partial stores (.cg) are net-positive. Combine-phase partial loads (.cg) is net-negative. Additional `.cg` knobs unlikely to yield wins.

## Verdict

**Reverted to exp_26 state.** Marginal consistent regression; cache-modifier axis done. Pivot required.

## Next directions

- **Pivot away from cache-modifier tuning.** exp_23/24 gave a few tenths of a percent; exp_28 costs the same back. Axis exhausted.
- **Structural candidates for exp_29**:
  - Retry compact-block partition with explicit int32 casts + bounds masking to fix exp_27's correctness bug.
  - BLOCK_N=256 on split path (halves high-valid iter count; risks low-valid regression but the prefix-valid workload mix favours small num_valid anyway — worth a test).
  - Sweep `num_stages` on the fused (T≤2) kernel — only exp_19 tuned split-kernel stages.
  - Research agent — it's been 3 experiments since exp_26 plan; due in 2-7 iterations per cadence rule.
