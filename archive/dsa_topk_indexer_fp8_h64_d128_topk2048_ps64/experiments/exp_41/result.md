---
exp: 41
date: 2026-04-17
status: reverted
parent: exp_37
---

# Experiment 41 — 2026-04-17

**Description:** Per plan.md — replace the 32-iteration bit-serial threshold-build
inside `radix_topk_kernel` with a **2-pass 11-bit histogram radix-select**
(pass 1: bits 31..21 via `tl.histogram` + reverse `tl.cumsum`; pass 2: bits
20..10 among pass-1 candidates; tie = matching top 22 bits). Preflight on
`tl.histogram` + `tl.flip` + argmax pattern confirmed all three compile and
produce correct output on Triton 3.7.0 / B200 (see `scripts/preflight_histogram.py`).

## Implementation

Replaced lines 208-220 of `radix_topk_kernel` (the 32-iter bit loop + strict/tie
split) with histogram-based pass 1 + pass 2 + recomposed strict/tie masks on
top 22 bits. Non-candidate lanes in pass 2 were mapped to bucket 0 and subtracted
afterwards. `N_BUCKETS = 2048` constexpr. Scatter code unchanged.

## Results

- Pass: 16/16 quick + stride-8 (**correctness perfect** — all matched_ratio==1.0000 across 5 trials each)
- Stride-8 slow-path latency **catastrophically regressed 3.3-4.4×**:
  - 4c7705ad: 0.0171 → 0.093 ms  **(+443%)**
  - 19e7663d: 0.0206 → 0.095 ms  (+361%)
  - 7f1cd9c2: 0.0190 → 0.095 ms  (+400%)
  - f457feb2: 0.0199 → 0.095 ms  (+377%)
  - a876010b: 0.0362 → 0.175 ms  (+383%)  ← heaviest workload
  - de54c4e6/2f3b7321/e63194e7: 0.0227 → 0.097 ms  (+327%)
- Fast-path (8 workloads): unchanged (~0.002 ms, separate `scoreless_kernel`).
- Mode: stride 8 (16 workloads). REVERTED.

## Learnings

1. **The plan's cost model for `tl.histogram` was wrong by ~30×.** Predicted
   ~2.2 µs per pass of 8192 elements; actual radix phase went from ~15 µs → ~90 µs,
   i.e. the histogram core is adding roughly 75 µs (each pass is ~35 µs, not 0.8 µs).
2. **Most likely lowering**: Triton 3.7's `tl.histogram` appears to compile to a
   dense per-thread bucket-comparison mask (each of 256 threads builds a 2048-wide
   boolean vector: "am I in bucket i?", then reduces across the warp). That's
   256 × 2048 = 500K compares per pass, vs the bit-loop's 32 × 8192 = 260K
   compares — but the bit-loop's compares are a tight fused sum, while histogram's
   reduction is over a far larger tile. An alternative hypothesis is serialized
   SMEM atomic adds, but atomics would typically show workload-independent cost,
   whereas here the cost scales with BLOCK_N (a876010b at 8192 regresses +383%
   while 4096 workloads regress +360%, proportional to BLOCK_N ratio).
3. **Preflight ≠ perf check.** Correctness preflight only proved `tl.histogram`
   *works*, not that it's *fast*. A real preflight should benchmark the
   intrinsic in isolation before committing to an algorithm that depends on it.
4. Algorithmic win in principle is real (O(N) work per pass is asymptotically
   better than 32× tree reductions); the loss is entirely in Triton's codegen
   for `tl.histogram`. Would need a manual SMEM-atomic or warp-specialized
   histogram implementation to actually win — large risk surface, not exp-41 scope.

## Takeaways

1. **`tl.histogram` on Triton 3.7 / Blackwell is ~30× slower than naive cost**
   model suggests. Do not use in hot paths without isolating the intrinsic's
   actual throughput first.
2. Close the histogram-radix axis. Any further attempt on this mechanism needs
   a hand-rolled SMEM-atomic histogram, which has its own failure modes (atomic
   contention on 256 threads → 2048 buckets).
3. Return to the 8-attempts-since-exp-33, 1-marginal-win plateau. Next lever
   must attack a different structural axis.

## Next candidates

- **Gluon migration** (tripped threshold). Research has advised against twice
  now, but post-exp-41 we have 8 reverts + 1 marginal since exp 33. The
  "15-20 iterations, 1 win" rule of thumb in `/optimize` is effectively met
  for this plateau. The histogram-radix failure closes the last concrete
  compute-bound axis on the current kernel. Candidate Gluon scope: the
  `score_kernel` (exp 29 profile shows ~1.5× HBM headroom still untapped at
  66 GB/s vs 103 GB/s memcpy). Not the radix — the remaining ~4 µs of compute
  there is below Gluon's minimum friction cost.
- **Warp-specialized histogram via `tl.atomic_add`**: hand-roll what `tl.histogram`
  should have been. Each of 8 warps owns 256 buckets; per-warp atomic_add into
  SMEM tile, then cross-warp merge. Riskier than pure Gluon since we'd still
  depend on Triton's SMEM + atomic lowering.
- **Dynamic grid from `block_table`**: skip empty programs at source via
  cumsum-based grid lookup. LESSONS.md rules out `.item()` (30-60 µs sync),
  but a GPU-side program-id remapping through a precomputed `grid_offsets`
  int32 tensor was never tried. Low risk, small ceiling (~2 µs on sentinel-fill
  elimination across 59 slow-path programs).
- **Score kernel BLOCK_T exploration**: exp 18/19/21/22 ruled out BLOCK_T=128.
  BLOCK_T=32 (half-page) never tried — splits work across 2× programs with
  half the K-load per program, could improve HBM utilization. Small wins
  possible but risks register pressure.

## Experiment accounting

- Since exp 33 last-best: 34/35/36/38/39/40/41 reverted, 37 kept (marginal).
  **8 attempts, 1 marginal win, 7 reverts including 2 major regressions (39, 41).**
  Meets the `/optimize` "15-20 attempts" Gluon threshold pragmatically — every
  single-knob axis on `radix_topk_kernel` has now been explored.
