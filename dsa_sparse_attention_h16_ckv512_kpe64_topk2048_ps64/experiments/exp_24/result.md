# Experiment 24 — 2026-04-17

**Description:** Add `cache_modifier=".cg"` on `partial_m`, `partial_l`, `partial_acc` STORES in `_fused_split_combine_kernel` split phase. Hypothesis: writes bypass L1 (per-CTA, pointless since combine CTAs read from L2), land directly in L2. Combine CTAs spin less waiting for L2 coherence of partials.

## Results
- Pass: 12/12 on stride-2 (quick: 2/2)
- Max abs err: 1.56e-02
- Mode: stride-2 + A/B paired

**A/B vs exp_23** (paired, same VM):
| UUID | T-class | A (exp_23) | B (exp_24) | Δ |
|---|---|---|---|---|
| 02d6ae9c | T=8 | 0.0155 ms | 0.0155 ms | -0.10% |
| 05f6de65 | T=2 | 0.0186 ms | 0.0186 ms | +0.19% |
| 0c23b10c | T=1 | 0.0055 ms | 0.0055 ms | +1.41% |
| 2207f0fd | T=8 | 0.0159 ms | 0.0159 ms | -0.06% |
| 232ed014 | T=8 | 0.0154 ms | 0.0153 ms | -0.38% |
| 4c46a94b | T=6 | 0.0154 ms | 0.0153 ms | -0.33% |
| 5096e459 | T=8 | 0.0159 ms | 0.0159 ms | -0.02% |
| 564007ac | T=8 | 0.0160 ms | 0.0160 ms | -0.08% |
| 78b2e11c | T=8 | 0.0155 ms | 0.0155 ms | -0.08% |
| b7668cfd | T=1 | 0.0054 ms | 0.0055 ms | +0.24% |
| e6b849f2 | T=2 | 0.0080 ms | 0.0080 ms | +0.04% |
| f77df5ce | T=2 | 0.0054 ms | 0.0054 ms | +0.36% |

Paired: 7/12 B wins, mean Δ = -0.0000 ms. **Large-T (T≥6): 7/7 B wins, Δ ∈ [-0.38%, -0.02%]**. Directionally consistent, still below noise floor.

## Design

Change:
```diff
- tl.store(pm_ptrs, m_i)
- tl.store(pl_ptrs, l_i)
- tl.store(pacc_ptrs, acc)
+ tl.store(pm_ptrs, m_i, cache_modifier=".cg")
+ tl.store(pl_ptrs, l_i, cache_modifier=".cg")
+ tl.store(pacc_ptrs, acc, cache_modifier=".cg")
```

Rationale: partial_m/l/acc are producer-consumer across CTAs within the same kernel call. The producer CTA writes, 8 different CTAs read each piece via the atomic-barrier sync. L1 retention per-CTA is zero benefit — the consumer is a different CTA whose L1 is cold. `.cg` pushes directly to L2 where the consumer will read.

The atomic barrier spin (lesson 25) relies on `tl.load(volatile=True)` to poll via L2. For partial_* data, the store-side cache modifier matters because the producer's `release` atomic only guarantees ordering, not how fast data reaches L2. With default `.wb` (write back), data may dwell in L1 before eviction; `.cg` commits directly to L2.

## Discoveries

1. **`cache_modifier=".cg"` works on `tl.store` in Triton 3.6.** Compiles and runs cleanly.
2. **Producer-consumer cross-CTA stores benefit from `.cg` on the producer side.** Consumer is a different CTA whose L1 is empty for the line; L1 retention at producer is waste. Change is zero-risk, trivial to revert.

## Verdict

**Marginal, kept.** Same pattern as exp_23: 7/7 large-T wins in consistent direction, but each <0.4%. Mean Δ ≈ 0. Keeping because it's zero-cost. Not a "new best" at stride 2 resolution — would need full benchmark to confirm a real ~0.1-0.2% mean win.

## Next directions

- **Combine kernel's partial_* LOADS with `.cg`** — symmetric change on the combine side. If partial_* is L2-resident, default .ca cache-all is fine; if .cg forces re-fetch from L2 (not memoized in L1 across iters), could be neutral or regress.
- **Try `.cs` on stores** — may be supported on stores even though it wasn't on loads (different PTX path). Streaming writes bypass both L1 and L2 retention, flushing directly to HBM. But combine immediately re-reads, so HBM round-trip would regress.
- **Pivot away from cache tuning.** Deltas are saturating below the noise floor. Bigger levers needed: inline-PTX `launch_dependents`, workload-specialized dispatch, or back to Gluon tcgen05_mma.
