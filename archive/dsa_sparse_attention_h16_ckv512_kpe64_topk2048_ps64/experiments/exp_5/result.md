# Experiment 5 — 2026-04-16

**Description:** Ablation of the combine-kernel change alone. Isolate whether switching combine from `tl.static_range(8)` + `num_stages=1` (exp_2) to dynamic `range(8)` + `num_stages=2` helps at the NUM_SPLITS=8 baseline (separating it from the NUM_SPLITS=16 change that regressed in exp_4).

Kernel: exp_2 + combine loop swap (static_range→range, num_stages 1→2).

## Results
- Pass: 12/12
- Kernel latency (ms): small=0.026 / large=0.030 / overall=0.026 (min) / 0.030 (median) / 0.031 (max)
- Reference latency (ms): not profiled
- Max abs err: 1.56e-02
- Mode: stride 2 (12 workloads); A/B confirmed against exp_2 on same VM

**A/B vs exp_2 (paired, same VM):**
| | A (exp_2) | B (exp_5) | Δ |
|---|---|---|---|
| small (T≤2) | 0.022-0.025 | 0.027-0.028 | +19-23% ❌ |
| large (T≥6) | 0.025 | 0.031-0.032 | +25-28% ❌ |
| All 12 workloads | A wins 12/12, mean Δ = +0.0057 ms (+22%) | | |

## Learnings
- **Combine-kernel change is harmful at NUM_SPLITS=8.** Dynamic loop + num_stages=2 over 8 iterations of 32-KB body adds ~5-7 µs per token. With T=1..8, that's the dominant fraction of the combine time.
- **`tl.static_range(8)` is optimal for combine at this size.** The unrolled body is small enough to fit registers, and Triton's compiler handles LDG issue ordering better with the static loop. `num_stages=2` on a dynamic loop forces async prefetch infrastructure that doesn't amortize over 8 iters.
- **The combine kernel is probably not the bottleneck** — if it were, the overhead would be proportionally smaller. The 22% regression in total latency suggests combine was already ~25-30% of exp_2 latency (so ~6-7 µs). Compute-bound split kernel at ~18 µs is the bigger target.

## Decision
**Revert to exp_2 state** (static_range combine + num_stages=1). Move on to a new optimization axis.

## Candidate axes for exp_6+
1. **Break-at-end-of-iteration early exit** (workload-specific; compute always, break at loop end on all-padding). 88% of tokens have >1024 padding → big opportunity. Lesson from exp_3 says `if`-prefix defeats pipelining, but break-at-end should be safe.
2. **BLOCK_N tuning** (32 or 128 instead of 64).
3. **num_warps / num_stages tuning on split kernel.**
4. **tf32x3 in dot** (already using default precision; try explicit).
5. **Tile-aligned page load** (contiguous-run fast path when all valid).
