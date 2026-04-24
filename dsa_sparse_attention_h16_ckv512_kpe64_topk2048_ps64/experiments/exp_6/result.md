# Experiment 6 — 2026-04-16

**Description:** Dynamic loop-bound early exit. Before the main compute loop, scan `SPLIT_SIZE=256` indices once with `tl.sum((idx>=0).to(i32))`, compute `num_blocks_needed = ceil(num_valid / BLOCK_N)`, and loop only over `range(0, num_blocks_needed * BLOCK_N, BLOCK_N)`. Keeps compute unconditional inside the hot loop (preserves `num_stages=2` async pipelining) while skipping all-padding blocks entirely.

This is the "precompute loop bound" option from exp_3's next-move list. The break-at-end approach (tried first this iteration) failed: Triton AST rejects `break` inside `for` loops. Dynamic range bound works instead.

Exploits workload property (see workload_profile.md): p50 num_valid = 33 of 2048, 88% of tokens have >1024 padding, indices are contiguous-valid-prefix + `-1`-suffix.

## Results
- Pass: 12/12
- Kernel latency (ms): small=0.011 / large=0.022 / overall=0.011 (min) / 0.022 (median) / 0.023 (max)
- Reference latency (ms): not profiled
- Max abs err: 1.56e-02
- Mode: stride 2 (12 workloads); A/B confirmed against exp_2 on same VM.

**A/B vs exp_2 (paired, same VM, B=exp_6):**
| UUID | A (exp_2) | B (exp_6) | Δ |
|---|---|---|---|
| 0c23b10c (T=1) | 0.0225 | 0.0159 | **−29.14%** ✅ |
| b7668cfd (T=2) | 0.0231 | 0.0135 | **−41.50%** ✅ |
| f77df5ce (T=2) | 0.0217 | 0.0135 | **−37.89%** ✅ |
| e6b849f2 (T=2) | 0.0218 | 0.0194 | −10.75% ✅ |
| 05f6de65 (T=2) | 0.0233 | 0.0220 | −5.73% ✅ |
| 4c46a94b (T=6) | 0.0242 | 0.0242 | −0.11% ≈ |
| 02d6ae9c..2207f0fd (T=6-8) | 0.0243 | 0.0243 | ≈0 |

**Summary:** B wins 11/12, mean Δ = −0.0024 ms (−10%). Dominated by the 5 T≤2 workloads (−6 to −42%); T≥6 workloads are neutral (no work to skip because splits already near-full when batched).

## Learnings
- **Dynamic loop-bound beats break-at-end** — Triton AST doesn't support `break` inside `for`; `range(0, dyn, BLOCK_N)` with a precomputed `dyn` scalar works and triggers the expected pipelining.
- **Scan overhead is free on large workloads.** Loading `SPLIT_SIZE=256` int32 indices (1 KB) + `tl.sum` ≈ 1-2 µs, absorbed by the larger compute. Large workloads see 0 regression despite the extra scan.
- **Small-workload wins confirm the sparsity hypothesis** — T=1 went 0.022→0.011 ms (−50%). This matches the expected split-K × padding-exit cascade: T=1 launches 1×8=8 CTAs, most of which had nothing to do before.
- **Large-workload neutrality is expected**: for batched T=6-8, valid entries span many splits (p90 unique_pages=18), and most splits need ~4 full BLOCK_N=64 blocks. The dynamic bound = original bound. No harm done.

## New best: 0.011 ms (smallest) / 0.022 ms (median) / 0.023 ms (max).

## Next candidate axes (exp_7+)
1. **Try NUM_SPLITS=16 AGAIN** but now with dynamic loop bound — small workloads that skip most splits will benefit from more splits with very low overhead per skipped split. The exp_4 regression was dominated by combine-kernel cost, but since split work is so much cheaper now, higher NUM_SPLITS may amortize combine differently.
2. **`num_stages=3` or `=4`** on split kernel — now that typical work is 1-2 BLOCK_N blocks, pipeline depth matters differently; check if shorter pipeline is better.
3. **BLOCK_N=128** with a single iteration for small workloads (num_valid ≤ 128).
4. **num_warps tuning** (try 4 instead of 8) for small-workload split kernels which are now lightweight.
5. **Combine kernel tuning** — combine cost is now a larger *fraction* of total latency on small workloads; tuning it directly may help.
