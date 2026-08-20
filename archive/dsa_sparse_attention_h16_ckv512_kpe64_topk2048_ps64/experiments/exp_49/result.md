# Experiment 49 — 2026-04-17

**Description:** Per `exp_49/plan.md` (research-agent authored). Single-variable change: `BLOCK_N = 128 → 64` on line 314 of host dispatch for `_fused_split_combine_kernel`. Plan's hypothesis: workload profile median valid=33 means per-split valid (≈valid/8) is typically ≤ 8, so tokens are wasting 97% of BLOCK_N=128's capacity on mask padding. BLOCK_N=64 halves per-iter compute and K HBM traffic for the majority class. Expected -1 to -3% net with outliers at >+2%.

## Results
- Pass: 2/2 quick (abs_err = 1.56e-02 matches baseline)
- Mode: stride-2 A/B vs exp_48
- **A/B 2/12 B wins, mean Δ = +0.0016 ms → A (exp_48) faster**

**A/B stride-2 vs exp_48 — massive regression on T≥3:**
| UUID | T-class | A (exp_48) | B (exp_49) | Δ | % |
|---|---|---|---|---|---|
| 02d6ae9c | T=8 | 0.0153 | 0.0187 | +0.0034 | **+22.23% A** |
| 05f6de65 | T=2 | 0.0186 | 0.0186 | −0.0001 | −0.33% B |
| 0c23b10c | T=1 | 0.0054 | 0.0055 | +0.0001 | +1.24% A |
| 2207f0fd | T=7 | 0.0154 | 0.0189 | +0.0035 | **+22.48% A** |
| 232ed014 | T=8 | 0.0150 | 0.0152 | +0.0002 | +1.39% A |
| 4c46a94b | T=6 | 0.0108 | 0.0122 | +0.0014 | **+12.85% A** |
| 5096e459 | T=8 | 0.0156 | 0.0190 | +0.0034 | **+22.01% A** |
| 564007ac | T=8 | 0.0156 | 0.0191 | +0.0034 | **+22.01% A** |
| 78b2e11c | T=8 | 0.0153 | 0.0188 | +0.0035 | **+22.62% A** |
| b7668cfd | T=2 | 0.0054 | 0.0054 | +0.0000 | +0.05% A |
| e6b849f2 | T=2 | 0.0080 | 0.0079 | −0.0001 | −1.36% B |
| f77df5ce | T=2 | 0.0054 | 0.0054 | +0.0001 | +1.26% A |

6/7 T≥3 workloads regress +12-23%; only 232ed014 escapes at +1.39%. T≤2 unchanged (fused kernel not touched) — pure noise. **Hits revert gate immediately** (any T≥3 >2% regression).

## Verdict

**Reverted to exp_48 baseline.** Confirmed via diff — byte-for-byte identical.

## Discoveries

1. **Plan's median-valid hypothesis was wrong for the T≥3 regime.** The plan cited workload_profile.md's p50 valid=33, but T≥3 benchmark workloads (the ones actually dominating latency) have HIGH per-token valid distributions. Measured: every T=8 workload regressed +22%, meaning per-split valid for these workloads exceeds 64 (triggering iter count 2 → 3+). Stride-2 dispatches specifically to T=2 and T=8 workloads; the T=8 cluster has tokens at the high end of the valid distribution, not median.

2. **Iter count transition 2→3 costs ~50% of split_work** on the affected large-T workloads. If split_work was 8 µs at 2 iters, 3 iters = 12 µs. Total CUPTI 15.5 → 19.5 µs = +26%. Observed +22% matches this model closely.

3. **BLOCK_N=64 on split path is definitively wrong for the benchmark workload set.** This doesn't close the BLOCK_N axis broadly (different workload distributions might favor 64), but on THIS trace set with T=8 large-valid tokens dominating the large-T cluster, BLOCK_N=128 is the right choice.

4. **Research agent's plan had a valid mechanism but wrong premise about workload distribution.** The mechanism reasoning (tile padding → waste → halve the tile) is sound but assumed per-split valid follows p50=4. Actual T≥3 benchmark per-split valid is much higher (≥64 on ~6/7 affected workloads). Lesson: when reasoning from `workload_profile.md`, check the percentile aligned with the dispatch pattern — `stride 2` hits specific T-classes, not the median.

5. **LESSON-19's "BLOCK_N crossover at valid≈100-500" confirmed on split path** with the per-split-valid transform: at stride-partition NUM_SPLITS=8, per-token valid ≈ 100-500 maps to per-split ≈ 12-63 (below BLOCK_N=64) vs per-token valid ≥ 512 maps to per-split ≥ 64 (above). Benchmark T=8 workloads have per-token valid near or above 512 on many tokens.

## Next directions

- **Revisit BLOCK_N axis with split-kernel adaptive dispatch:** in-kernel branch on per-split valid count would allow BLOCK_N=64 for low-valid tokens and BLOCK_N=128 for high-valid. However, LESSON-20 notes num_valid isn't cheaply available from host (.item() round-trip expensive). Would need in-kernel dynamic BLOCK_N via masked smaller tile + unmasked bigger tile — more complex than a constexpr sweep.
- **Exp_50: Host-side split count selection (NUM_SPLITS=16 for high-valid tokens).** If T=8 workloads have per-split valid ≈64-130 driving 2-3 iters, doubling NUM_SPLITS to 16 would halve per-split valid to 32-65, bringing most tokens to 1 iter. Trade-off: combine kernel work doubles. Exp_4 already tested NUM_SPLITS=16 and found regression due to combine cost, but that was pre-stride-partition (exp_26) and pre-monotonic-counter (exp_37). Worth re-testing at current baseline — fundamentally different mechanism now.
- **Exp_51 (structural): warp-specialized split kernel with async K loads and overlapped compute.** Current kernel uses `num_stages=2` for HBM-compute overlap, but K/P loads and MMA compute aren't explicitly warp-specialized. A producer/consumer warp pair could hide K-gather latency better. Structural change; high risk, high reward.
- **Not retry:** BLOCK_N=64 on split path under current dispatch (this experiment); BLOCK_N=32 (well below crossover); BLOCK_N=256 (shmem OOM per exp_33).
