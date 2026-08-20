# Plan — exp 23

## Diagnosis

Three consecutive same-axis experiments on the **mp=2 fast path** axis failed
at similar magnitudes and produced nearly indistinguishable latencies:
- exp 18 (BLOCK_T=128 big-MMA in *score_kernel*): +0.5 µs mean, medium workloads
  regressed +5-7%.
- exp 21 (BLOCK_T=128 big-MMA in 1-program-per-batch *fused* kernel for mp=2):
  mp=2 workloads regressed from ~25 µs to ~70 µs (+180%). +1.6 µs mean.
- exp 22 (BLOCK_T=128 two-dot fp32-combine in 1-program-per-batch *fused*
  kernel for mp=2): **identical** 67-69 µs latency to exp 21, +1.7 µs mean.

Exp 22 was a clean negative control for the fp8-SHMEM-shuffle hypothesis —
it eliminated the 16 KB fp8 tile rearrangement and still regressed the same
amount. The overhead is therefore **not** in fp8 layout conversion. The shared
remaining factors between exp 21 and 22 are: BLOCK_T=128, `tl.sort` on 128
uint64 elements, register pressure from simultaneous 2-page live state, and
loading two K pages per program. Three data points (exp 18 on score_kernel,
exp 21 on fused big-MMA, exp 22 on fused two-dot) all at BLOCK_T=128 land
near the same per-program cost on mp=2 targets.

**Repetition loop detected**: the mp=2 fast-path idea has been tried in two
distinct structural forms and regressed identically. A third mp=2 structural
variant is unlikely to teach us something new, and the ceiling is small.

**Ceiling analysis:**
- **mp=2 fast path** (this axis, 3 tries failed): 5 mp=2 workloads × ~15 µs
  max save = ~0.6 µs mean. Extending to mp∈{3,4} would add 6 workloads at
  similar ceilings, total ~1.9 µs. BLOCK_T=64 doesn't apply (would need two
  programs per batch, defeating fusion). BLOCK_T=128 is demonstrably bad on
  this problem. Ceiling effectively capped.
- **Alloc elimination for `torch.empty((B, max_scored))`** (untried on THIS
  axis): 9 µs dispatch on every non-fast-path call; 61/128 workloads have
  `max_scored ≤ 2048`, so can alias into `topk_indices`. Ceiling: 9 × 61/128
  ≈ **4.3 µs mean**.
- **Flat (sum_active,) grid** (workload_profile.md Opt #2, untried): 1-3 µs
  per call × 128 workloads. Ceiling: 1-3 µs mean.
- **torch.topk replacement** (partially attempted in exp 8, 15): large ceiling
  but high-risk; requires a radix-select or threshold-scan implementation
  not in single-iteration scope. Deferred.

**Pivot target**: alloc elimination has the highest ceiling of the
single-iteration-feasible changes, a clean scope, and is orthogonal to all
prior experiments. Exp 16's lesson explicitly directs this way: "*If alloc
overhead is 9 µs, you need to eliminate the `torch.empty` call itself (e.g.,
pass in a pre-allocated buffer as an argument), not layer a dict on top of
it.*" A Python-level dict cache won't work; aliasing the output tensor
(which the kernel already receives as DPS) as the scratch buffer avoids the
`torch.empty` dispatch entirely.

## Strategy

**Pivot**. Abandon the mp=2 fast-path axis — ceiling is ~0.5-1.9 µs mean and
three independent attempts (exp 18/21/22) failed. Pivot to **alloc
elimination via DPS output aliasing**, where the ceiling is ~4.3 µs mean and
the change is orthogonal to all previous work.

## Actions (priority ordered)

1. **What:** In `solution/triton/indexer_fused.py` host dispatch, replace
   `scores = torch.empty((batch_size, max_scored), ...)` with a **zero-copy
   dtype reinterpretation of the DPS output tensor** when `max_scored ≤ topk`:
   ```python
   max_scored = max_num_pages * page_size
   if max_scored <= topk:  # 2048
       # Alias topk_indices (int32 [B, 2048]) as fp32 scratch — same
       # storage, no torch.empty dispatch. score_kernel overwrites it with
       # fp32 scores; torch.topk reads it as fp32; remap_kernel then
       # overwrites topk_indices with the final int32 indices. PyTorch
       # default-stream ordering guarantees read-before-write safety.
       scores = topk_indices.view(torch.float32)
   else:
       # mp > 32: max_scored > 2048 doesn't fit in topk_indices. Fall back.
       scores = torch.empty((batch_size, max_scored), device=q_index_fp8.device,
                            dtype=torch.float32)
   ```
   Leave `stride_sb = scores.stride(0)` (= 2048 in aliased case, = max_scored
   in allocated case) — `score_kernel` already parameterizes on the stride.
   Leave `torch.topk(scores, effective_topk, dim=-1)` — reads fp32, unchanged.

   **Why:** `profile.md` shows `torch.empty` is 9.4-9.8 µs flat on B200 — torch
   dispatch machinery, not allocator work (exp 16 ruled out allocator pooling).
   The DPS output `topk_indices` is a 2048-int32-per-batch buffer **already
   allocated by the caller**. For `max_scored ≤ 2048` it has enough storage
   to hold the fp32 scores; reinterpreting via `.view(torch.float32)` is a
   ~0.5 µs CPU op (exp 17) vs 9 µs for `torch.empty`. The kernel later
   overwrites the buffer with the real int32 indices via `remap_kernel`, which
   is the only "downstream writer" after `torch.topk` finishes consuming the
   scores. Default-stream implicit ordering guarantees
   `score_kernel → torch.topk → remap_kernel` serialization.

   **Impact:** 9 µs saved on 61/128 workloads = **~4.3 µs mean ceiling**
   (~9% of current mean 49.6 µs). Fast-path (mp=1) workloads unaffected
   (8 workloads stay on exp 20 path). mp>32 workloads unaffected (remain on
   allocated path). No correctness risk expected — same fp32 compute, same
   int32 output.

2. **What:** Validate correctness aggressively — run `modal run
   scripts/run_modal.py --quick` first (smallest + largest workload; catches
   any shape assumption). The smallest is likely mp=1 (fast path, unchanged);
   the largest is mp=91 (allocated path, unchanged). Neither exercises the
   aliased-buffer path! Therefore explicitly run `modal run
   scripts/run_modal.py --stride 16` next (16 workloads) to exercise mp∈[2,32]
   workloads that use the new aliased path.

   **Why:** Quick covers mp=1 and mp=91 — both fall OUTSIDE the new code path
   (aliased buffer). A clean quick-run won't prove correctness of the change.
   We need at least one mp∈[2,32] workload in the sample.

   **Impact:** Eliminates false-positive "quick passed → ship" risk.

3. **What:** If stride-16 passes 16/16, run the full 128-workload benchmark +
   `scripts/ab_benchmark.py::run --a experiments/exp_20/indexer_fused.py` for
   paired signal. Expected paired Δ: -2 to -5 µs mean (many A/B samples
   hit the mp∈[2,32] regime).

   **Why:** Absolute mean Δ of 4 µs is below cross-VM noise floor (per
   CLAUDE.md: "For sub-5% deltas vs previous best, use scripts/ab_benchmark.py").
   Paired A/B is the only reliable signal for this magnitude.

   **Impact:** Confirms or rejects the ceiling estimate.

## Do not try

- **Another mp=2 fast-path variant** (exp 18/21/22 all regressed). In
  particular:
  - BLOCK_T=128 with any internal structure (big-MMA, two-dot, tl.join
    combine) — all tested, all ~70 µs per program on 1-program-per-batch
    grids. The compiler/Triton codegen has a per-program cliff at BLOCK_T=128
    that 1-SM occupancy cannot hide.
  - BLOCK_T=64 with two programs per batch fused into one kernel — defeats
    the fusion benefit (now `grid=(B, 2)`, same as default `score_kernel`
    except with sort/output-write added; strictly slower than just making
    score_kernel faster).
  - mp=3/mp=4 extensions via the same structure — at larger N the register
    pressure and sort-at-larger-BLOCK_N cost get worse, not better.
- **Module-level dict cache for `scores`** (exp 16 tied; dispatch dominates,
  dict.get adds ~0.3 µs that washes the saving).
- **Skipping `torch.as_strided` or merging views** (exp 17 tied; py_setup
  floor is noise).
- **BLOCK_K change in remap_kernel** (exp 12 regressed +0.8% uniformly).
- **num_warps=8 on score_kernel** (exp 11 regressed smallest +12.8%).
- **`.item()` sync for dynamic effective_topk** (exp 13/14 both regressed).
- **`tl.sort` on BLOCK_N ≥ 4096** (exp 8/15; hits the bitonic wall).

## Coordination notes

- **This is one atomic change** — swap the `torch.empty` call for a
  conditional `.view()`. No coupled kernel edits. Suitable for single-run
  validation.
- **Verify quick-mode covers the new code path**: the smallest workload is
  mp=1 (doesn't hit the new path) and the largest is mp=91 (doesn't hit the
  new path). **A passing `--quick` run does NOT prove correctness**. Must
  run `--stride 16` next to exercise mp∈[2,32] workloads that actually use
  the aliased buffer.
- **A/B vs exp 20** (not exp 10) — exp 20 is current best (has the mp=1
  fast path which this change preserves).
- **No profile needed before coding**: the cost (`torch.empty` at 9 µs) is
  already measured in `profile.md`; the fix is direct.
- **If regressed**: revert immediately. Possible failure modes are (a)
  torch.topk's dtype/stride check rejects the aliased fp32 tensor (would
  show as a run-time error on the first aliased workload), (b) implicit
  stream ordering is violated and we get a data race (would show as
  correctness failure, not perf), or (c) torch.empty was actually cheaper
  than profile.md suggested (would show as ≤0.5 µs mean improvement,
  i.e., washed). Each of these is diagnosable in one run.

## Optional follow-up (not this experiment)

If exp 23 wins, the next candidate is **Opt #2 (flat-grid score_kernel)** from
`workload_profile.md`, which addresses the 70.9% wasted-program problem. If
exp 23 ties, re-read profile.md alloc numbers with fresh CUDA-event-isolated
measurement — may indicate profile.md overcounted.
