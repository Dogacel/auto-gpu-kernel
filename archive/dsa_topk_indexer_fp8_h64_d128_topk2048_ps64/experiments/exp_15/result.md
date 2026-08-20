---
exp: 15
date: 2026-04-17
status: reverted
parent: exp_10
---

# Result — Tile-merge triton top-K (reverted)

## Change
Replaced `torch.topk` with a Triton `topk_kernel` using tile-merge
selection. Per batch: initialize a [TOPK=2048] uint64 running key to
zeros, then for each BLOCK_CHUNK=2048-sized chunk of scores, pack
(monotone_bits, inv_idx) into uint64, concat with running key into
[4096], sort descending, take first TOPK as new running key. Decode
indices from final running key.

Tried two concat variants:
1. `tl.cat(running, new, can_reorder=True)` — **compile error**:
   `TritonGPURemoveLayoutConversions` pass failure in MLIR pipeline.
2. `tl.join` + `tl.trans` + `tl.reshape` — compiles and passes
   correctness (matched_ratio=1.0), but severely regresses perf.

## Measurement

`--quick` (exp 15 variant 2):
- 30cecff1 (smallest): **0.131 ms** (exp 10 was 0.025 ms = +106 µs / +524%)
- dba1e960 (largest): **0.116 ms** (exp 10 was 0.060 ms = +56 µs / +93%)

Never ran A/B — quick regressed enough to kill the idea.

## Why it lost

1. **[4096] sort is slow on B200.** Triton's bitonic sort is
   O(N log² N); at N=4096, log²=144 vs 121 at N=2048 (+19% stages
   theoretical, but 5× in practice due to register / layout pressure).
   Exp 8 already measured a 27-349% regression vs torch.topk at BLOCK_N ≥ 4096.
2. **NUM_CHUNKS=1 workloads still pay the full 4096 sort.** Even when
   max_scored ≤ 2048 (50% of workloads), the tile-merge structure still
   sorts [2048 zeros + 2048 real scores] = [4096]. No early-exit possible
   without specialization.
3. **NUM_CHUNKS≥2 workloads would pay it per chunk** (2-3 sorts of
   [4096] each). On a876010b (NUM_CHUNKS=3), this would be ~60-90 µs of
   sort time — worse than torch.topk's 51 µs.

## Lesson

Triton's `tl.sort` hits a hard performance wall at BLOCK_N ≥ 4096 on
B200 that **tile-merge cannot hide**, because each merge step requires
a [2*TOPK] sort of the concatenated running + new buffers. The
theoretical O(N log² N) + register pressure blow up past 2048 and make
even small workloads (that could have used a single [2048] sort)
regress badly.

**Options that remain for beating torch.topk:**
- A **branched** fallback (`if NUM_CHUNKS==1: sort [2048] else: torch.topk`)
  per exp 8 V2 — already tried, Python branch overhead absorbs wins.
- A **radix-select** kernel (O(N) passes over scores, 4-pass for int32
  partitioning). Much more complex, out of one-iteration scope.
- **Reduce the work feeding into torch.topk**: shrink scored region so
  torch.topk has less to sort. `.item()`-based dynamic K already ruled
  out (exp 13, 14). No viable path remains here.

Given the 5× regression on the smallest workload and the known
scaling wall at BLOCK_N ≥ 4096, **revert to exp 10** and seek
improvements outside the torch.topk phase.

## Reverted to exp 10 state

All tile-merge / topk_kernel removed. Proceed to exp 16 on a different
axis — score_kernel register/tile tuning, scores allocation reduction,
or similar. torch.topk is effectively a local optimum we cannot
beat with bitonic-sort-based Triton kernels at this K.
