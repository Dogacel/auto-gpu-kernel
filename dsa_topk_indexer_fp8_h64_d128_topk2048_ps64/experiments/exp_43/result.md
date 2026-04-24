---
exp: 43
date: 2026-04-17
status: kept (marginal)
parent: exp_37
---

# Experiment 43 — 2026-04-17

**Description:** Convert `score_kernel`'s Q and K loads from raw pointer
arithmetic (`tl.load(base + offs)`) to `tl.make_block_ptr` + `tl.load(bp)`.
This is the canonical Triton idiom for communicating block shape/stride/order
metadata to the compiler, which on Blackwell Triton 3.7 can enable TMA-lowered
async loads for large contiguous tiles.

Only Q (`[BLOCK_H, BLOCK_D] = [64, 128]` FP8) and K (`[BLOCK_T, BLOCK_D] = [64, 128]`
FP8) — the two 8 KB HBM loads per program. The smaller scalar loads (`scale[64]`,
`w[64]`, `block_table`) stay as raw-pointer loads because the metadata overhead
outweighs the vectorization benefit at 256 B.

## Implementation

Replaced lines 114-118 of `score_kernel` with `tl.make_block_ptr` constructors
and `tl.load(bp)` calls. No logic change. `order=(1, 0)` declares D-axis as the
contiguous inner dim.

## Results

- Pass: quick 2/2, matched_ratio=1.0 across 5 trials each. Both workloads
  are fast-path (not exercising score_kernel) — relied on A/B for signal.
- A/B vs exp 37 (stride 8, paired same-VM):
  - **Run 1**: B wins 10/16, mean Δ -0.00003 ms. Slow-path 5/8 B wins
    (-0.08% to -0.83%); a876010b -0.21%.
  - **Run 2**: B wins 10/16, mean Δ -0.00004 ms. Slow-path 6/8 B wins
    (-0.13% to -1.00%); a876010b -0.13%; only +0.15% loss (f457feb2).
- Same pattern as exp 37 (kept at -0.0001 ms marginal): consistent
  directional signal on slow-path, fast-path within noise.
- Mode: A/B stride 8 (confirmed across 2 runs).

## Learnings

1. `tl.make_block_ptr` is worth trying for large HBM loads in compute-heavy
   kernels even when the baseline raw-pointer version is well-hoisted.
   Block pointers carry compile-time shape + runtime strides + contiguity order,
   which the Triton compiler uses to select vectorized load instructions and
   on Blackwell can prefer TMA over plain cp.async. Raw-pointer loads with
   runtime-valued strides may miss this path.
2. The gain here (~-0.4% slow-path mean) is small because the raw-pointer
   version was already well-compiled — it's not a structural mechanism, it's
   a hint that nudges the compiler toward a slightly better vectorization.
3. **The gain does NOT stack with future Gluon TMA attempts.** If we migrate
   score_kernel to Gluon with explicit `tl.tma.load`, the block-ptr benefit
   is absorbed into the Gluon path. For now, it's a small Triton-native win
   on top of exp 37.
4. No correctness surprise — block-ptr semantics are equivalent to the
   raw-pointer version for in-bounds loads (shape == block_shape means no
   masking is needed).

## Takeaways

1. **New best** (marginal, exp-37-style). A/B mean Δ is within noise but
   2× consistent slow-path lean + directional a876010b win justifies keeping.
2. The scalar loads (scale, w, block_table) do NOT benefit from block pointers
   — 256 B is below the vectorization threshold and the constructor overhead
   would likely wash it out.

## Next candidates

- **Hand-rolled Gluon score_kernel with TMA + tcgen05_mma** — still the highest-
  ceiling untried lever (exp 42 closed Gluon auto-translation; hand-roll is
  the remaining path). Ceiling: ~1.4 µs full-run mean from the HBM headroom
  analysis in profile.md. Requires a fresh-context sub-agent.
- **`num_ctas=2` on score_kernel** — Blackwell supports thread block clusters
  with distributed shared memory. Could enable cooperative loads between paired
  programs. Untried. Risk: may not compile cleanly.
- **Radix micro-axes** remain closed (exps 33-38 + 41).

## Experiment accounting

- Since exp 33 last-best: 34/35/36/38/39/40/41/42 reverted, 37 kept (marginal),
  **43 kept (marginal)**. 10 attempts, 2 marginal wins (exp 37 + 43), 8 reverts
  including 2 major regressions. Finally broke the 7-revert streak since exp 40.
