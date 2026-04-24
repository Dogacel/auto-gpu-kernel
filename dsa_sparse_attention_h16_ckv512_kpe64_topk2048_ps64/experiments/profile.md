# Kernel Profile
_Generated 2026-04-17 against `solution/triton/sparse_fused.py` @ cb13132 (exp_51 baseline: NUM_SPLITS=16, D_CKV_SPLIT=16, BLOCK_D=32)._

Harness: `scripts/profile_kernel7.py`. Amortized Event-pair p50 over 200 inner × 50 outer. Phases via stubbed subtraction. Numbers from run 2 (noop=7.28 µs; run 1 VM was +0.5 µs drifted).

## Headline (T=8, highest-valid workload, total_valid=4758, per-token max=2048)

- Full fused kernel p50 = **23.90 µs** harness; CUPTI (bench.log) = 0.0125 ms = 12.5 µs.
- Noop=7.28, prologue=11.45, Q-load=4.17. T=6/7/8 near-identical structurally.

## Phase breakdown, T=8 (p50 µs)

| Phase | exp_48 (NS=8) | exp_51 (NS=16) | Δ | % of full |
|---|---|---|---|---|
| Launch (noop) | 7.51 | 7.28 | -0.23 | 30% |
| Q load | 4.26 | 4.17 | -0.09 | 17% |
| **Split work** | 9.27 | 9.33 | **+0.06** | **39%** |
| **Combine work** | 5.18 | 5.12 | -0.06 | 21% |
| Barrier | 0.85 | 0.65 | -0.20 | 3% |

T=6/7: split_work=9.33/9.51, combine_work=5.20/5.30, barrier=0.77/0.69.

**Critical finding:** Doubling NUM_SPLITS 8→16 **did NOT reduce split_work in the harness** (flat at 9.3 µs). Yet CUPTI benchmark shows -2.7 µs wins on T=7/8. Discrepancy: 200-inner-iter harness saturates L2 + register pressure at 128 CTAs; per-CTA iters halved (2→1 BLOCK_N) but CTA-count doubled. CUPTI single-call captures the parallel fill-of-SMs win that the tight loop masks. Combine similarly held flat (5.18→5.12): smaller tile per CTA (BLOCK_D 64→32) but 2× static_range iters cancelled the gain.

## Observed vs memory floor (T=8, 5.35 MB K-gather)

- K memcpy floor at 5.48 MB = **7.08 µs**.
- Split work = **9.33 µs**; gap = 2.25 µs (up from exp_48's 1.39 µs because memcpy floor dropped ~0.8 µs on this VM). Still K-bandwidth-bound but with 2.25 µs of MMA/softmax/register overhead above floor.
- partial_acc memcpy = 4.60 µs (bytes unchanged: T×NS×H×D_CKV×4 preserves when D_CKV_SPLIT doubles and BLOCK_D halves).

## Hotspots

From exp_51 bench.log: **05f6de65 (T=8 max-valid) stays at 18.7 µs unchanged** — 2-iter even at NS=16. **4c46a94b (T=6 low-valid) regressed 10.8→12.4 µs** — already 1-iter at NS=8, doubling splits added combine overhead without split gain.

## Bottleneck

**Phase:** Combine (now 21% of budget and flat despite half-tile). **µs:** 5.12 (vs split 9.33). Split's HBM-bound status is unchanged but further reducible only via fewer K-bytes (structural). Combine has the clearest new lever:

**Lever:** fused 2-stage combine tree (16→4 first pass, 4→1 second) would cut combine to ~3 µs by eliminating the doubled `static_range(NUM_SPLITS)` loop overhead. Alternative: Q-load (4.17 µs, 17%) via cross-CTA shared-mem prolog or vectorized `ld.global.ca.128` — currently each of 16 CTAs re-fetches identical 18 KB Q.

**Ceiling if combine → 2-stage tree:** ~21.9 µs harness / ~10.5 µs CUPTI on T=7/8. Independent of adaptive NUM_SPLITS (exp_52) which addresses 4c46a94b regression orthogonally.

**Invalidated claim:** exp_48 profile's "within 1 µs of HBM floor" was NUM_SPLITS=8-specific; floor is partition-dependent. Split_work above-floor gap grew from 1.39 → 2.25 µs, meaning more compute overhead is now exposed at the new partition — another lever if combine is saturated.
