---
exp: 27
parent: exp_26
hypothesis: "Per-batch scoreless decision inside score_kernel + adaptive_remap_kernel
unlocks scoreless for the 96.9% of batch items with seq_len < 2048 that live in mp > 32
workloads (set-based matched_ratio still holds per batch)."
---

# Experiment 27 plan

## Observation

Exp 26 extended scoreless to `max_num_pages ≤ 32` because that guarantees
`seq_len ≤ 2048 = topk` for EVERY batch in the workload. Remaining 59 workloads
at `mp > 32` pay ~57.6 µs mean. But per workload_profile.md:

- **96.9% of batch items have seq_len < 2048** (only 3.1% = 59 items total need scoring)
- Mean intra-batch skew = 1181× — most batches in mp > 32 workloads are still tiny
- Only ~59 batch items across all 128 workloads actually have seq_len ≥ 2048

The insight from exp 25/26 (set-based matched_ratio) applies PER-BATCH:
if `seq_len[b] ≤ topk`, the scoreless natural-order output is correct for batch b,
regardless of what other batches in the workload look like.

## Plan

1. **`score_kernel` per-batch early-exit**: if `seq_len[b] ≤ topk`, write -1e30
   across this program's tile and return. Skips the MMA + sum + scale work for
   scoreless batches. The -1e30 writes keep `torch.topk` well-behaved (no NaN,
   tied scores are benign).
2. **Rename remap_kernel → `adaptive_remap_kernel`**: add a per-batch branch.
   If `seq_len[b] ≤ topk`, emit natural-order token IDs (same as scoreless_kernel).
   Else, use `topk_idx` path (the old remap logic).
3. `torch.topk` unchanged — still runs over the full [B, max_scored], but its
   result is DISCARDED for scoreless batches (adaptive_remap branch ignores it).
   This is OK: torch.topk handles all-tied -1e30 just fine (returns some permutation
   of valid indices; they're never read).

## Why this is safe

- Non-scoreless batches (3.1% of items): identical code path to exp 26. No correctness risk.
- Scoreless batches: natural-order output is the set of all valid tokens. Since
  seq_len ≤ topk, ALL valid tokens fit in the top-K set. Under set-based
  matched_ratio, any permutation of the set scores 1.0. ✓
- The score_kernel still writes valid -1e30 scores (not uninitialized memory),
  so torch.topk receives well-defined inputs.

## Expected outcome

Per-batch savings estimate for mp > 32 path:
- score_kernel was ~26-34 µs. Early-exit for scoreless batches skips ALL Q·K MMA
  and sum work. Just writes fp32 -1e30 (trivial memory BW).
- torch.topk unchanged (~50 µs)
- adaptive_remap: +small branch overhead vs remap
- Net per workload: ~20-25 µs saved on score_kernel work
- Slow-path mean: 57.6 → ~35 µs (−40%)
- Full mean: 0.0276 → ~0.018 ms (−35%)

## Risks

1. Branch divergence in score_kernel: if batches mix scoreless + non-scoreless,
   programs diverge on the `if seq_len <= topk` check. Triton handles this via
   warp-level divergence; programs belonging to different batches don't share
   warps, so this should be fine.
2. torch.topk with all-tied -1e30 values: produces valid but arbitrary indices.
   Adaptive_remap ignores them — no correctness impact.
3. Memory ordering: `score_kernel` writes then `torch.topk` reads — already
   serialized by stream semantics. No new race condition.

## Ablation / fallback

If correctness fails on quick benchmark, revert. Unlikely given exp 25/26 results.
If performance regresses (shouldn't — strict subset of work), investigate the
adaptive_remap branching cost and revert.
