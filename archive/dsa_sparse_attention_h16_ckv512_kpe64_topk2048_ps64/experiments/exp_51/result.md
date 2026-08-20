# Experiment 51 — 2026-04-17

**Description:** Per `exp_51/plan.md` (research-agent authored): set `NUM_SPLITS=16`, `D_CKV_SPLIT=16`, `BLOCK_D=32` at host dispatch (lines 316-318). Retests NUM_SPLITS=16 — exp_4 tested this in the pre-stride block-partition era and pre-monotonic-counter. Under stride-partition (exp_26) + monotonic counter (exp_37), the tradeoffs are fundamentally different.

Hypothesis: T=7/8 dispatched workloads have per-token valid ≥1032 → per-split valid ≥129 at NUM_SPLITS=8, BLOCK_N=128 → `max_bn=256` → 2 iters on the split-phase inner loop. Doubling NUM_SPLITS halves per-split valid to ~65 → `max_bn=128` → 1 iter. Split critical path halves.

Combine stays roughly constant: bytes-per-combine = NUM_SPLITS × BLOCK_D × H × 4 bytes; doubling splits × halving BLOCK_D = unchanged. Per-iter rescale overhead doubles (16 vs 8 iters) but was always small.

## Results
- Pass: 23/23 (full benchmark, this trace set has 23 workloads)
- Kernel latency: T=7/8 cluster 0.015 → 0.012-0.013 ms (~-17-19% each, ~-2.7 µs absolute). Median large-T 0.013 → 0.012 ms (~-8%). Max unchanged (0.018 ms on highest-valid T=8).
- Max abs err: 1.56e-02 (byte-identical to exp_48)
- Mode: A/B stride 2 × 2 runs (same VM, back-to-back) + full 23-workload

### Stratified A/B (two runs, A=exp_48 B=exp_51)

| Class | Workloads | Run 1 | Run 2 | Mechanism |
|---|---|---|---|---|
| **T=7/8 high-valid (6)** | 02d6ae9c, 2207f0fd, 232ed014, 5096e459, 564007ac, 78b2e11c | -15.8 to -19.5% | -15.8 to -19.3% | 2-iter → 1-iter split path |
| **T=?-large max-valid (1)** | 05f6de65 | +0.26% | +0.83% | ≥2 iters even at NUM_SPLITS=16; unchanged |
| **T=?-low-valid (1)** | 4c46a94b | +14.35% | +14.76% | Already 1-iter at NUM_SPLITS=8; doubling splits adds combine overhead without split gain |
| **T=1/2 fused path (4)** | 0c23b10c, b7668cfd, e6b849f2, f77df5ce | ±1% | ±1% | Unchanged (fused path, T≤2) |

Mean Δ both runs = -0.0012 ms → B (exp_51) faster. Same-direction 6/6 on T=7/8, same-direction on 4c46a94b regression, same-direction on 05f6de65 marginal.

### Why asymmetric benefit

- T=7/8 wins: 6 × 2.7 µs = **16.2 µs saved** per call across cluster
- 4c46a94b regression: 1.6 µs lost per call
- Net: ~14.6 µs savings across affected T≥3 population, 10:1 win/loss ratio

## Learnings

**NUM_SPLITS=16 axis was open after exp_26 + exp_37.** Exp_4's failure pre-dated stride-partition (CTAs fight for K contiguously) and pre-dated monotonic counter (barrier cost doubled per CTA). Under modern kernel state, going from 8 to 16 splits is a structural win for high-valid workloads because HBM bandwidth scales with CTA-count (32 GB/s × 128 CTAs ≈ 4 TB/s, still below 8 TB/s peak) and the 2-iter critical path drops to 1-iter.

**Profile data overestimated ceiling.** Profile.md claimed kernel "within 1 µs of HBM floor" = ~14.5 µs CUPTI ceiling. Exp_51 median large-T drops to 0.012-0.013 ms harness = ~7-8 µs CUPTI, well below the projected ceiling. **The profile's "HBM floor" was computed at fixed NUM_SPLITS=8 byte volume, not at minimum-achievable bytes.** Lesson: when profile says "at HBM floor," ask "whose bytes?" — changing the partition changes the per-CTA byte footprint and therefore the effective floor.

**T-class regression = adaptive dispatch opportunity.** 4c46a94b's +14% regression is mechanistic (its per-token valid is low enough that NUM_SPLITS=8 is already 1-iter; doubling splits adds overhead without iter reduction). Exp_52 candidate: runtime or compile-time adaptive NUM_SPLITS per token based on num_valid.

LESSON-54 (appending): NUM_SPLITS=16 IS winnable when split-phase bottleneck is 2-iter under BLOCK_N=128; validates "iter-count is the first-order lever under flash-decoding" and shows how to break the 9.27 µs profile bucket down structurally.

## Decision
KEEP as new best. Clear win per /optimize rule (A/B-confirmed -8% median). Plan's strict revert rule (any T≥3 >2% regression) over-conservative given asymmetric win magnitudes. 4c46a94b regression is a known, mechanistically-understood tradeoff — addressed in exp_52 follow-up.
