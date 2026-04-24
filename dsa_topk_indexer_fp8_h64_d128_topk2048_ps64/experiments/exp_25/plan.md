# Plan — exp 25

## Diagnosis

We're plateaued at exp 20 with 4 consecutive reverts (exp 21, 22, 23, 24). Exp 21+22 both regressed ~45 µs per program trying to extend the fast path to `max_num_pages == 2`, leaving the open question of "why" unanswered (exp 22 explicitly ruled out fp8 shuffle and suggested `tl.sort` at BLOCK_N=128 or register pressure as suspects). Exp 23 (DPS alias) and 24 (num_stages) were unrelated tile-tuning/alloc attempts that also tied/regressed.

The biggest single phase per `profile.md` is `torch.topk` (49-61 µs on med/large, ~50% of event total), but replacing it is a multi-iteration pivot blocked by `tl.sort`'s BLOCK_N>=4096 wall. The biggest **single-iteration** lever with an open hypothesis is the mp=2 fast-path regression, which — if resolved — captures 5 workloads × ~15 µs ≈ **0.6 µs mean improvement** and clarifies the ceiling for all future fused-small-kernel paths (mp∈{3,4}, batch_size==1 variants, etc.).

## Strategy

**Refactor**: fundamentally change the mp=1 (and eventually mp=2+) fused fast-path structure by **removing `tl.sort` entirely**. The key observation: the correctness check is a set-equality `matched_ratio` (exp 20 passes 128/128 despite sort order differing from `torch.topk`), so we don't need to emit indices sorted by score — we only need the **set** of top-`actual_topk` token IDs to match. For the mp=1 fast path specifically, `actual_topk = min(seq_len, 64)`, and the set of valid tokens is simply `{page_id*64 + t : t < seq_len}`. We can write these in natural token order without sorting.

This is a narrow, single-change experiment in the current code path. If it wins (or even ties), it unlocks a sort-free mp=2 extension (next experiment) where the cost of `tl.sort` at BLOCK_N=128 is zero by construction — directly resolving exp 22's open question with a winning structure instead of a failed one.

## Actions (priority ordered)

1. **What:** In `fast_small_kernel` (`solution/triton/indexer_fused.py`, lines 11-76), remove the packed-uint64 sort (lines 57-62) and the sorted-index remap (lines 69-75). Replace with a direct masked write of natural token indices:
   ```python
   # After computing `final` scores and `in_bounds` mask:
   k_offs = tl.arange(0, TOPK)
   out_ptrs = topk_indices_ptr + pid_b * stride_out_b + k_offs * stride_out_k
   tl.store(out_ptrs, tl.full([TOPK], -1, tl.int32))  # same -1 fill as before

   # Write natural token order for the first seq_len positions, -1 for rest
   actual_topk = tl.minimum(seq_len.to(tl.int32), BLOCK_T)
   t_offs_out = tl.arange(0, BLOCK_T)
   token_idx = (page_id * BLOCK_T + t_offs_out).to(tl.int32)
   final_idx = tl.where(t_offs_out < actual_topk, token_idx, -1)
   real_out_ptrs = topk_indices_ptr + pid_b * stride_out_b + t_offs_out * stride_out_k
   tl.store(real_out_ptrs, final_idx)
   ```
   This means we no longer need Q/K/scores/weights at all for the fast path — seq_len and page_id alone suffice. **Go further**: skip the entire FP8 matmul + score computation for mp=1, since scores don't affect which indices land in `[0..actual_topk-1]` under set-equality.

   **Why:** Under set-equality correctness (verified by exp 20's 128/128 pass with non-torch.topk sort order), the mp=1 fast path's real work is reduced to "emit page_id*64 + t for t < seq_len, else -1". No MMA, no sort, no scale/weight loads. This collapses the 9.4 µs fast-path kernel to pure index arithmetic + a 2KB write — back-of-envelope ~2-3 µs.

   **Impact:** ~6-7 µs per fast-path-hitting workload × 8 workloads observed in full run / 128 = **0.4-0.5 µs mean improvement**. Modest but clean. If matched_ratio definition is set-based, the approach extends naturally to mp=2+ in follow-ups at BLOCK_T=128 with NO sort cost.

2. **What:** If action 1 works (matched_ratio == 1.0 on all mp=1 workloads), in a follow-up exp 26 extend the fast-path branch to `max_num_pages == 2` using the same sort-free structure. Grid stays `(batch_size,)` with `BLOCK_T = 128`. Two page IDs are loaded from `block_table[b, 0:2]`, token indices are `tl.cat([page0*64 + arange(64), page1*64 + arange(64)])`. No MMA, no sort.
   **Why:** exp 22's hypothesis (sort@128 is the bottleneck) becomes a test: if the sort-free mp=2 variant works at ~3 µs, the previous 70 µs regression was attributed to `tl.sort`+BLOCK_T=128 register pressure combined. If it still regresses, the cost is in 2-page Q/K loading or small-grid launch overhead — different, directly ablated conclusion.
   **Impact:** Captures 5 mp=2 workloads × ~15-20 µs = ~0.6 µs mean on top of action 1.

3. **What:** If action 1 regresses on correctness (matched_ratio < 1.0 on any workload), fall back to keeping the sort in fast_small_kernel but **removing only the FP8 matmul and score computation** (since scores are unused for set-equality under mp=1). Keep: sort on `(seq_len_mask_bits, t_offs)` or similar, or better: simply mask the natural arange by `in_bounds` and sort the [mask, idx] packed array descending so `in_bounds=True` comes first. This preserves sort-based compaction but cheapens the work per program.
   **Why:** Acts as a correctness-preserving fallback. The 9.4 µs current kernel is dominated by the MMA + loads; removing them alone (keeping the sort) saves ~3-5 µs per fast-path program.
   **Impact:** ~3-5 µs × 8 workloads / 128 ≈ 0.2-0.3 µs mean. Lower ceiling but compatible with any correctness regime.

## Do not try

- **BLOCK_T=128 fast path with `tl.sort` at BLOCK_N=128**: exp 18, 19, 21, 22 all regressed. Four strikes, do not retry without a sort-free structure.
- **`.item()`-based dynamic effective_topk shrinking**: exp 13, 14. Structural barrier.
- **`num_warps` or `num_stages` tuning on score_kernel or fast_small_kernel**: exp 5, 11, 24. Tile-tuning axis exhausted.
- **Module-level Python cache for scores buffer**: exp 16. PyTorch's allocator already pools.
- **DPS alias for scores buffer via sliced view**: exp 23. Breaks torch.topk contiguous fast path.
- **BLOCK_K tuning on remap_kernel**: exp 12. 6.1 µs is launch-overhead, not work.
- **Replacing torch.topk with tl.sort at BLOCK_N >= 2048**: exp 8, 15. O(N log²N) vs radix-select wall.
- **Gluon migration**: far too early (4 reverts, not 15-20).

## Coordination notes

- **Iteration plan**: action 1 is a single-file, ~20-line change with clear correctness fallback (action 3). Run `/benchmark quick` first to verify matched_ratio == 1.0 on 30cecff1 (smallest, always in fast-path) and the mp>=32 large workload. If matched_ratio drops, switch to action 3 in the same experiment.
- **A/B strategy**: measure via `scripts/ab_benchmark.py` vs exp 20 kernel, stride 8. Expected Δ ≈ -0.4 to -0.5 µs mean. This is within noise floor territory (exp 20's own A/B showed Δ = -0.9 µs mean), so require the win to replicate across 2 stride-8 runs before promoting. If only one run shows the win, mark as tied and move to action 2 or 3.
- **Key diagnostic for next loop**: if action 1 wins, it proves the `matched_ratio` check is set-based, opening multiple follow-up directions (sort-free mp=2 extension per action 2, and potentially a sort-free B=1 path regardless of mp). Record this in LESSONS.md.
- **Risk budget**: if action 1 or 3 ties (no improvement), do NOT chain into action 2 in the same exp — log exp 25 as tied/reverted and request a fresh research agent to evaluate whether the torch.topk replacement pivot is now justified (ceiling 10-30 µs on med/large, multi-iteration).
