# Experiment 31 — 2026-04-17

**Description:** Halve `NUM_SPLITS` from 8 to 4 (coupled `D_CKV_SPLIT = 4`, `BLOCK_D = 128`) under stride-partition. Hypothesis: combine-phase knobs were tuned under block-partition (exp_3-5); stride-partition might shift the optimum. Fewer splits = less atomic contention + halved combine work.

## Results
- Pass: 2/2 quick (T=1 unchanged; T=8 correctness PASSED but latency **regressed 62%**)
- Mode: quick only — did not proceed to A/B
- **Reverted immediately** — correctness ok, perf regression too large

**Quick data:**
| UUID | exp_26 (NUM_SPLITS=8) | exp_31 (NUM_SPLITS=4) | Δ |
|---|---|---|---|
| 0c23b10c (T=1) | 0.005 ms | 0.005 ms | — (small-T path) |
| 2207f0fd (T=8) | 0.016 ms | 0.026 ms | +62% |

## Design (reverted)

```diff
-    D_CKV_SPLIT = 8
-    NUM_SPLITS = 8
+    D_CKV_SPLIT = 4
+    NUM_SPLITS = 4
```

## Discoveries

1. **NUM_SPLITS=8 is a strict optimum** for both block- and stride-partition, not just block. exp_4 under block-partition regressed going 8→16; exp_31 regresses going 8→4. The sweet spot holds across partition schemes.

2. **Root cause of regression**: under stride-partition, each split's valid-entry count is `num_valid_total / NUM_SPLITS`. Going NUM_SPLITS=4 doubles per-split valid count → doubles per-split iterations for high-valid workloads. For T=8 workload 2207f0fd (which has high valid count), this means up to 4 iters/split instead of 2. The split critical path grows linearly with iters, while combine savings are only ~30-50% of a small phase. Net ~60% regression.

3. **The combine phase is too small to benefit from halving**. Per `profile.md`, combine is ~12-15% of total time. Even halving it saves ~6-7%. Meanwhile, the split phase critical path doubles on high-N → 2× the bottleneck. Math doesn't work.

4. **NUM_SPLITS re-tune under stride is closed.** Both directions (4 and 16) regress. 8 is optimal.

## Verdict

**Reverted to exp_26.** Fast quick-mode result; no A/B needed.

## Next directions

- **NUM_SPLITS axis conclusively exhausted** across both partition schemes (block and stride).
- Remaining candidates that are still fresh:
  1. **Re-examine fused (T≤2) kernel**: BLOCK_N_FUSED=64 was tuned (exp_13) but never swept down to 32 or up to 128 under the current D_CKV_SPLIT=8 small-T path.
  2. **num_stages=1 on split kernel**: untested; most workloads only 1 iter under stride so num_stages=2 pipelining may be unused overhead.
  3. **Gluon rewrite with Blackwell primitives** (`bw.tcgen05_mma` + `bw.mbarrier`) — per skill guidance, after ~15-20 plateau iterations. Currently the plateau around exp_26 is 5 revert-iterations deep; worth considering but commit to 5-10 Gluon iterations runway.
  4. **Workload-specialized dispatch**: host-side separate tiny kernel that writes per-token valid counts, main kernel uses for better branch scheduling. Adds launch overhead; marginal case only.
- Lean toward (2) for exp_32 — cheapest test with single-knob change. Then pivot to Gluon for exp_33+ if (2) is flat.
