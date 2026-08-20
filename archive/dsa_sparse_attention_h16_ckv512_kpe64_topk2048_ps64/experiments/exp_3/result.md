# Experiment 3 — 2026-04-16

**Description:** Added early-exit `if tl.max(idx) >= 0:` around the entire per-block compute (load K/V + dots + online softmax). Workload profile showed median token uses 33 of 2048 valid entries — most BLOCK_N blocks are fully padded. Hypothesis: skip the work on padded blocks → large speedup especially for T=1, T=2.

NUM_SPLITS=8 unchanged from exp_2 to isolate the early-exit change.

## Results
- Pass: 12/12
- Kernel latency (ms): small=0.013 / large=0.030 / overall=0.013 (min) / 0.029 (median) / 0.030 (max)
- Reference latency (ms): not profiled
- Max abs err: 1.56e-02
- Mode: stride 2 (12 workloads)
- vs exp_2: **mixed** — small workloads −33% to −38% (0.021→0.013), large workloads **+25%** (0.024→0.030) → net regression on median.

**Per-workload deltas:**
| UUID | exp_2 | exp_3 | Δ |
|---|---|---|---|
| 0c23b10c (T=1) | 0.021 | 0.013 | −38% ✅ |
| b7668cfd | 0.022 | 0.015 | −32% ✅ |
| 05f6de65 | 0.023 | 0.029 | +26% ❌ |
| e6b849f2 | 0.022 | 0.019 | −14% ✅ |
| f77df5ce | 0.021 | 0.014 | −33% ✅ |
| 4c46a94b | 0.024 | 0.029 | +21% ❌ |
| 02d6ae9c..2207f0fd (batched T=6-8) | 0.024 | 0.030 | +25% ❌ |

## Learnings
- **The `if` around the whole compute defeats software pipelining.** With `num_stages=2`, Triton issues async loads for the next iteration while computing the current. A branch that gates the compute prevents the compiler from issuing those speculative loads → large workloads (where the `if` is always true) pay an overhead penalty from lost pipelining.
- **Sparse-token wins are real** (−33% to −38% on T=1/T=2) but the price on batched workloads (+21-25%) outweighs the win at median.
- **Next move (exp_4):** keep the early-exit signal, but restructure so the pipeline stays intact. Options:
  - **Break-at-end-of-iteration**: always compute this block (cheap when masked-out with `valid` guard), then `break` if `max(idx) < 0`. "Full" workloads never break, no overhead. Sparse workloads exit after first all-padding block.
  - **Precompute loop bound**: scan the split's indices once, bound the loop by `num_valid_blocks`.

Revert exp_3 and try break-at-end for exp_4.

## Sub-lesson for LESSONS.md
Adding a scalar-conditioned `if` inside the main hot loop breaks `num_stages=N` prefetching: the Triton compiler can't issue async K/V loads for iteration i+1 when iteration i's compute is conditional. Use structured patterns that keep the compute unconditional (break-at-end, or mask the work with a `tl.where` pattern).
