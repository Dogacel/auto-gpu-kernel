---
exp: 27
date: 2026-04-17
status: reverted
parent: exp_26
---

# Experiment 27 — 2026-04-17

**Description:** Per-batch scoreless short-circuit inside `score_kernel` plus a
new `adaptive_remap_kernel` that branches per batch on `seq_len <= topk`.
Goal: skip the MMA+sum work for the 96.9% of batch items with seq_len<2048 that
live inside `mp > 32` workloads (where they currently pay full scoring).

## Implementation

1. `score_kernel`: before computing the MMA, load `seq_len` and check
   `seq_len <= topk`. If so, write -1e30 to all `BLOCK_T` positions and `return`.
2. `adaptive_remap_kernel` (replaces `remap_kernel`): per-batch branch. If
   `seq_len <= topk`, emit natural-order token IDs. Else, use `topk_idx`
   result as before.
3. `torch.topk` unchanged — still runs on the full [B, max_scored] tensor.
   Its output for scoreless batches is tied -1e30s → harmless garbage indices
   that adaptive_remap discards.

## Results

**`/benchmark quick`:** 2/2 passed (fast-path workloads unchanged).

**`/benchmark stride 8`:** 16/16 passed.
Per-workload latencies indistinguishable from exp 26.

**A/B vs exp 26 (stride 8, paired):**
- B wins 3/16, mean Δ = -0.0000 ms → **tied**
- Largest B wins: 19e7663d (-0.76%), a876010b (-0.32%), f457feb2 (-0.08%) — all noise
- Largest A wins: e49574dd (+0.80%), df80c00b (+0.32%) — all noise

## Decision: Reverted

Zero measurable gain. Reverted to exp 26 state.

## Why it didn't work

The hypothesis assumed `score_kernel` MMA work was a meaningful slice of
slow-path latency. It isn't, for **parallelism reasons**:

- `score_kernel` launches B × mp programs on a 132-SM B200. For mp ∈ [33..91]
  and B ∈ [1..30], 33 to 2700 programs. Most workloads fit in 1-2 waves.
- **Wall-clock time = slowest program time, not sum.** The existing per-program
  `token_start >= seq_len` early-exit already handles inactive pages in the same
  wave as active ones. GPU parallelism hides the MMA cost within the wave.
- Killing the MMA for whole batches doesn't shorten the critical path when the
  wave already contains a non-early-exit program running MMA.

`torch.topk` (~49-61 µs per profile.md) dominates slow-path latency. We didn't
touch it.

## Lessons

1. **Parallel kernels aren't bandwidth-sum; they're max-per-wave.** Removing
   MMA work from programs that were already in-flight alongside active ones
   saves nothing. The `token_start >= seq_len` early-exit (exp 9) worked because
   it eliminated whole waves when mp far exceeded seq_len; this exp's per-batch
   check just swaps MMA for -1e30 stores within the same wave.
2. **Set-based matched_ratio unlock doesn't automatically translate to
   slow-path wins.** It requires a way to bypass torch.topk, which still runs
   over the padded [B, max_scored] tensor regardless of per-batch logic inside
   the remap.
3. **The next direction must attack torch.topk itself or kernel-level fusion
   that absorbs torch.topk.** Per-program early-exit has reached its ceiling.

## Next candidate

Targets for exp 28: the slow path (59 workloads, ~57.6 µs mean). Options:

- **Option A**: Triton custom top-K that handles all-scoreless batches as a
  no-op AND does actual selection only for scoring batches. One kernel instead
  of {empty, score, topk, remap} four-op pipeline. High risk (exp 8, 15 failed
  on tl.sort at BLOCK_N ≥ 4096) but potentially 20+ µs on slow path.
- **Option B**: Reduce `torch.topk` input size. Requires GPU→host sync on
  `max(seq_lens)` — prior attempts (exp 13, 14) cost ~60 µs. Net loss.
- **Option C**: Fuse `score_kernel` + `remap_kernel` + a bitonic top-K into
  one Triton kernel using reused SRAM. Complex.
- **Option D**: Profile the slow path in detail (CUDA events) before choosing,
  to confirm torch.topk is really the dominant slice.

Leaning D → then A or C. D is one iteration and will dictate which structural
change is worth the risk.
