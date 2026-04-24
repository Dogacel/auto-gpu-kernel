# Experiment 2 — 2026-04-16

**Description:** Flash-decoding split-K. Split TopK=2048 across `NUM_SPLITS=8` programs → grid=(T, 8) for phase-1 instead of (T,) → 8× more SMs active per token. Phase-2 combine kernel does a small online reduction across splits per token.

Partial workspace (fp32, per call): `partial_m [T,S,H]`, `partial_l [T,S,H]`, `partial_acc [T,S,H,D_CKV]`. Phase-1 writes unnormalized `acc` (not divided by `l`) so combine does a clean two-variable rescale.

**Bug fix (critical):** splits whose every index is `-1` produced `m_i = -inf`. Then `m_new = max(-inf, -inf) = -inf` and `exp2(m_i - m_new) = exp2(-inf - -inf) = NaN` → NaN propagated to output. Fixed with `m_new_safe = where(m_new == -inf, 0.0, m_new)` and `alpha = where(m_i == -inf, 0.0, exp2(m_i - m_new_safe))`. Same guard added to combine kernel. Note: exp_1 worked without this because in a single-program kernel over 2048 indices, all-invalid is implausible; with 8 splits of 256 each, a fully-padded split becomes common.

Tile: same `BLOCK_N=64`. Phase-1 `num_warps=8, num_stages=2`. Phase-2 `num_warps=4, num_stages=1`.

## Results
- Pass: 12/12
- Kernel latency (ms): small=0.021 / large=0.024 / overall=0.021 (min) / 0.024 (median) / 0.024 (max)
- Reference latency (ms): not profiled
- Max abs err: 1.56e-02  |  Max rel err: 7.89e+03 (abs err matches exp_1 → numerical parity maintained)
- Mode: stride 2 (12 workloads)
- vs exp_1: 0.100 → 0.024 ms, **~4.2× speedup**, new best

## Learnings
- **Biggest structural win available at this stage.** A 1-CTA-per-token kernel uses only `num_tokens` SMs on B200 (148 available). Splitting the TopK reduction by 8× multiplies program count and unlocks massive parallelism. The 4× latency drop likely under-states the headroom — with NUM_SPLITS tuned per num_tokens we may go further.
- **`exp2(-inf - -inf) = NaN` is a landmine.** Any kernel that runs online softmax over a subset that might be fully-padded needs the `where(m == -inf, 0.0, ...)` guard on `m_new`, `m_i`, and subsequent weights. Worth adding to LESSONS.
- NUM_SPLITS=8 chosen as a middle ground; 4 iterations per split. Small workloads clump at 0.021-0.022 ms, large at 0.024 ms → small/large gap has shrunk from 0.018 ms (exp_1) to 0.003 ms. Small workloads now have room for more splits (16?); large workloads already close to their min.
- Two-kernel launch overhead is ~5μs × 2 ≈ 10μs (hardware), visible in the 0.020+ ms floor. Fusing combine into phase-1's last block is a possible follow-up.
