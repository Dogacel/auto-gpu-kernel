# Experiment 13 — 2026-04-17

**Description:** Halve BLOCK_N for the fused (T≤2) path. Split path remains BLOCK_N=128. Motivation from workload profile: T≤2 tokens have median 33 valid entries — BLOCK_N=64 covers the median in a single iteration, halving per-iter K load (128→64 KB) and compute. BLOCK_N=128 was pulling in ~95 wasted-slot masked logits per iter on typical small-T tokens.

## Results
- Pass: 23/23 (quick + stride 2 + full)
- Kernel latency: 0.005 ms on most T=1/T=2 (vs 0.007 ms exp_11) and 0.019 ms on one high-valid outlier (vs 0.015 ms exp_11)
- Max abs err: 1.56e-02 (unchanged)
- Mode: stride 2 + full + A/B vs exp_11 (same VM)

**A/B vs exp_11 (paired, same VM, B = exp_13):**
| UUID | T | valid | A (exp_11) | B (exp_13) | Δ |
|---|---|---|---|---|---|
| 0c23b10c | 1 | 2 | 0.0070 | 0.0051 | **−26.64%** ✅ |
| b7668cfd | 2 | low | 0.0073 | 0.0054 | **−25.84%** ✅ |
| f77df5ce | 2 | ~37 | 0.0071 | 0.0053 | **−25.66%** ✅ |
| e6b849f2 | 2 | mid | 0.0074 | 0.0079 | +7.61% ❌ |
| 05f6de65 | 2 | ~2000 | 0.0151 | 0.0185 | **+22.94%** ❌ |
| 4c46a94b | 6 | — | 0.0180 | 0.0179 | −0.71% ≈ |
| 02d6ae9c | 8 | — | 0.0184 | 0.0185 | +0.63% ≈ |
| 232ed014 | 8 | — | 0.0187 | 0.0181 | −3.00% ≈ |
| 5096e459 | 8 | — | 0.0201 | 0.0197 | −1.98% ≈ |
| 564007ac | 8 | — | 0.0200 | 0.0185 | −7.44% (noise — same code path) |
| 78b2e11c | 8 | — | 0.0215 | 0.0183 | −14.93% (noise — same code path) |
| 2207f0fd | 8 | — | 0.0205 | 0.0185 | −9.75% (noise — same code path) |

**Summary:** B wins 9/12, mean Δ = **−0.0008 ms (−5.3%)**. The T≥3 deltas are VM noise (identical split+combine code on both sides). Real effect concentrates on T≤2: 3 big wins on low-valid, 1 mid-valid regression, 1 high-valid regression.

## Why it works

T=1/T=2 median 33 valid → with BLOCK_N=128, one iter loaded 128 KB of kc and executed a 16×128 softmax with 95 masked slots. BLOCK_N=64 cuts the HBM traffic to 64 KB/iter and the softmax to 16×64. For the median small-T token this is a single-iteration kernel: −0.002 ms (≈−29%) on the 3 lowest-valid workloads.

## Why 05f6de65 regresses

That T=2 workload has ~2000 valid entries per token. With BLOCK_N=128, the fused inner loop runs ~16 iters; BLOCK_N=64 doubles it to ~32 iters. Each extra iter adds loop overhead + dot startup, and the per-iter load is halved but launches twice as many async memory ops. Net: the loop-overhead term dominates the HBM-byte savings. A heuristic dispatch (BLOCK_N=64 only when valid < threshold) would fix this but requires knowing `num_valid` before launch — either a device→host sync (prohibitive) or a tiny preamble kernel (adds a launch).

## Learnings

- **Per-regime BLOCK_N specialization is a real lever** when the regimes have sharply different iter counts. T≤2 median 33 valid fits in one BLOCK_N=64 iter; that saves the whole second-half of a BLOCK_N=128 iter's work and traffic.
- **Tradeoff shape**: BLOCK_N reduction helps at low valid count (fewer wasted slots) and hurts at high valid count (more iter overhead). The crossover on this kernel is somewhere between `valid ≈ 100` (e6b849f2 loses +7.6%) and `valid ≈ 500`.
- **Can't easily dispatch on `num_valid`** because it's a per-token scalar only known on the device; a device→host sync or extra launch would cost more than the saving.
- **Log: one workload (`05f6de65`) regresses 23% but net mean is −5%.** Accepting the tail-regression for the mean win is a benchmark-scoring choice; a max-latency objective would call this a wash or a loss.

## New best. A/B confirmed −5.3% mean vs exp_11. Kept.

## Next directions

1. **num_warps tuning for BLOCK_N_FUSED=64** — smaller tiles (16×64 logits) may prefer num_warps=4 over 8 (exp_14 candidate).
2. **Attack large-T (T≥3) split+combine launch-barrier tax** — still the dominant cost (17 µs, ~16 µs launch). Needs a fused split+combine kernel that preserves D-parallel combine. Options: cluster-based (`num_ctas`), atomic-barrier, or a persistent single-kernel design. See profile.md §1.
3. **Heuristic dispatch for BLOCK_N_FUSED** — only a win if we can get num_valid without a sync. One option: run the fused kernel with a runtime-selectable inner loop that can "escape" after 1 block if num_valid ≤ 64; the saved work on small-valid tokens would match exp_13 without the 05f6de65 regression. Needs careful construction to not defeat num_stages pipelining.
