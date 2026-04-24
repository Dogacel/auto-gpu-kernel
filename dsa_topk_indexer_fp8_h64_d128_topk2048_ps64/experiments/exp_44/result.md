---
exp: 44
date: 2026-04-17
status: reverted
parent: exp_43
---

# Experiment 44 — 2026-04-17

**Description:** Extend exp 43's `tl.make_block_ptr` idiom from the 2D Q/K
loads in `score_kernel` down to the 1D scores load in `radix_topk_kernel`.
Hypothesis: if block-ptr helped the 8 KB Q/K loads, it might also help the
16-32 KB fp32 scores load in the radix phase, which is the next-largest
HBM load in the kernel.

## Implementation

Replaced the masked raw-pointer load of `scores[pid_b, :BLOCK_N]` with a
`tl.make_block_ptr` + `tl.load(bp, boundary_check=(0,), padding_option="zero")`.
`padding_option="zero"` is semantically equivalent to `other=float('-inf')`
here because the OOB slots are immediately re-masked to 0 by the subsequent
`tl.where(in_bounds, mono, 0)` step that converts fp32 → monotone uint32.

## Results

- Pass: quick 2/2.
- A/B vs exp 43 (stride 8, paired same-VM):
  - 5/16 B wins, mean Δ = +0.0000 ms.
  - Slow-path: 5/8 A wins. Three slow-path workloads clearly regressed:
    +0.62% (7f1cd9c2), +0.78% (de54c4e6), +1.17% (a876010b).
  - No A/B-confirmed B wins.
- Mode: A/B stride 8.

## Learnings

1. **Block-ptr doesn't generalize to 1D scalar-stride loads.** The Q and K
   loads in score_kernel are 2D tiles (`[64, 128]` FP8 = 8 KB each) with
   well-defined row/col strides and contiguity metadata — that's the exact
   shape where TMA and vectorized LDSM benefit from the block-ptr hint.
   The scores load is 1D fp32 with a single stride (unit-stride after the
   batch offset), so there is no extra contiguity metadata for the compiler
   to exploit. The raw-pointer masked version already compiles to a single
   coalesced LDG.128.
2. **Block-ptr overhead is non-zero.** `boundary_check` + `padding_option`
   have to emit runtime guards even when the boundary is statically trivial.
   On 16-32 KB loads this overhead swings small slow-path workloads +1%.
3. **The exp 43 marginal win was load-shape-specific, not a general idiom.**
   Block-ptr helps *large 2D tile loads*, not all HBM loads.

## Takeaways

1. Reverted. Live kernel restored to exp 43 state.
2. Do NOT generalize block-ptr to other raw-pointer loads without verifying
   each one is a large-2D-tile case. The scale, w, block_table, and scores
   loads all stay as raw pointers.
3. Closes the "micro-tune block_ptr to more loads" axis. The only remaining
   high-ceiling lever is hand-rolled Gluon TMA + tcgen05_mma on score_kernel
   (exp 42 closed the auto-translator path).

## Next candidates

- **Hand-rolled Gluon score_kernel with TMA + tcgen05_mma** — fresh-context
  sub-agent. Ceiling: ~1.4 µs full-run mean from profile.md HBM analysis.
- **`num_ctas=2` on score_kernel** — Blackwell thread-block clusters with
  distributed shared memory. Untried.
- **`num_ctas=2` / `num_warps` tuning** via autotune sweep on score_kernel.

## Experiment accounting

- Since exp 33: 34/35/36/38/39/40/41/42/44 reverted, 37 + 43 kept marginal.
  11 attempts, 2 marginal wins. Need a structural lever next — current
  micro-tune axis is tapped.
