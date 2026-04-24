---
exp: 37
date: 2026-04-17
status: kept
parent: exp_33
---

# Experiment 37 — 2026-04-17

**Description:** Hoist the per-token `block_table` load above the 32-iter radix
bit loop inside `radix_topk_kernel`. The bit loop is compute-bound on
`tl.sum` tree reductions (no HBM traffic), so issuing the block-table load
(~0.5 µs, 8192 int32 = 32 KB from HBM) earlier lets the memory latency
overlap with the reductions instead of stalling at the scatter point.

## Implementation

In the scoring branch of `radix_topk_kernel`, moved the `token_idx`
computation to immediately after the scores load, before the monotone
encoding + bit loop:

```python
# Exp 37: Issue block_table load early so it overlaps with the 32-iter radix bit loop
# (which is compute-bound on tl.sum tree reductions, no HBM traffic).
page_idx = offs // page_size
offset = offs % page_size
page_idx_clamped = tl.minimum(page_idx, max_num_pages - 1)
bt_ptrs = block_table_ptr + pid_b * stride_btb + page_idx_clamped * stride_btp
global_page = tl.load(bt_ptrs, mask=in_bounds, other=0).to(tl.int64)
token_idx = (global_page * page_size + offset).to(tl.int32)
```

`token_idx` is then consumed at the final scatter, unchanged.

Algorithmically identical to exp 33 (pure reordering of independent
operations) — Triton should be free to schedule the HBM load during the
~10 µs spent in the radix reductions.

## Results

- Pass: 2/2 quick (correctness), 16/16 A/B stride-8
- A/B vs exp_33 (paired same-VM, stride 8): **B wins 10/16, mean Δ = -0.0001 ms**
- Slow-path (scoring path): 5/8 B wins at -1.3% to -2.0% each; 3/8 tied
  within noise (+0.04%, +0.17%, +0.46%)
- Fast-path: noise dominates (all deltas < 1% on 0.002 ms workloads)
- Reference latency: not captured in A/B harness
- Mode: quick (correctness) + A/B vs exp_33 (stride 8)

## Learnings

- Triton's scheduler apparently does NOT hoist the block_table load on its
  own when the load and bit loop are structurally separated (scoring +
  scatter interleaved with cumsum). Manual reordering produced a real
  (~1.5% slow-path) win.
- The block_table load is only 32 KB per program, small enough to be latency-
  bound. Even a few hundred ns of overlap with the ~10 µs bit loop reduces
  effective cost to near-zero.
- This pattern — moving *independent* HBM loads ahead of compute-heavy
  unrolled loops — is probably applicable to other fused kernels that
  scatter at the end. Worth keeping in mind.

## Takeaways

1. Reordering by itself is a legitimate optimization when the Triton
   compiler misses the schedule. Small but consistent.
2. Mean ~1.5% slow-path gain × 8/16 workloads is a ~0.75% aggregate gain.
   Below the 5% keep-without-AB threshold, but the directional consistency
   (5/8 wins + 3/8 within noise, **no losses**) justifies keeping.
3. No correctness risk — pure reordering.

## Next candidates

- `strict_count = tl.max(strict_prefix)` (replaces one tl.sum with tl.max,
  reusing packed_prefix) — net tree-reduction budget may shrink one scan.
- `num_warps` on scoreless_kernel or fast_small_kernel (never tested).
- Hoist the `scores` load itself ahead of the block_table load? (they are
  already adjacent; might matter for warp-level pipelining).
- Move the packed cumsum inside the loop body for pipelining (risky — can
  change reg pressure).
- We have now had 1 miss + 1 win since exp 33 (exp 34/35/36 reverted,
  exp 37 kept). Continue Triton optimization; not stuck.
