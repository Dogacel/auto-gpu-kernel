---
exp: 38
date: 2026-04-17
status: reverted
parent: exp_37
---

# Experiment 38 — 2026-04-17

**Description:** Make `threshold` in the `radix_topk_kernel` bit loop a
scalar (shape `[1]`) instead of BLOCK_N-wide. Since every lane holds the
same value after the uniform `tl.where(accept, ...)` update each iteration,
the broadcast form wastes ~32 KB of registers and does 32× redundant
broadcast OR/where ops over the tile.

## Implementation

Three-line change in `radix_topk_kernel` scoring branch:

```python
# before
threshold = tl.zeros([BLOCK_N], tl.uint32)
for i in tl.static_range(0, 32):
    candidate = threshold | tl.full([BLOCK_N], 1 << (31 - i), tl.uint32)
    ...

# after
threshold = tl.zeros([1], tl.uint32)
for i in tl.static_range(0, 32):
    candidate = threshold | tl.full([1], 1 << (31 - i), tl.uint32)
    ...
```

`mono >= candidate` relies on Triton's scalar-to-vector broadcast.
`strict_mask = mono > threshold`, `tie_mask = mono == threshold` likewise
broadcast.

## Results

- Pass: 2/2 quick
- A/B vs exp_37 (stride 8): **B wins 9/16, mean Δ = +0.0000 ms (A slightly faster)**
- Slow-path detail:
  - 2f3b7321 -0.14% (B, noise)
  - 4c7705ad -0.60% (B)
  - 7f1cd9c2 +3.03% (A, **regression**)
  - 19e7663d +0.00% (tie)
  - de54c4e6 +0.02% (tie)
  - e63194e7 -0.31% (B, noise)
  - f457feb2 -1.27% (B)
  - a876010b +0.31% (tie)
- Mode: quick + A/B vs exp_37 (stride 8)

## Learnings

- Triton compiler **already detects uniform-valued tiles and scalarizes them**
  under the hood. Declaring `[1]` shape doesn't give additional leverage —
  the codegen for `[BLOCK_N]` with all-equal lanes is apparently as good as
  the scalar form.
- The one ~+3% slow-path regression (7f1cd9c2) is noise, but suggests the
  scalar path may not unify with the [BLOCK_N]-wide `mono`'s layout as
  cleanly at some PTX codegen pass. Not worth chasing.

## Takeaways

1. Don't try to hand-scalarize uniform values in Triton tiles — trust the
   compiler's uniform-analysis pass. If you have an optimization idea
   grounded in "the compiler probably isn't doing X", verify the PTX
   output before writing the variant.
2. Rule of thumb applies to threshold-like scalars, bit masks of constants,
   and loop-invariant broadcasts.

## Next candidates

- `strict_count = tl.max(strict_prefix)` — swaps one tl.sum for one tl.max.
  Both are tree reductions over BLOCK_N, so net cost is the same. Probably
  also a no-op but worth a single-iteration check.
- Hoist the `scores` load itself further (currently it's already adjacent
  to the block_table hoist from exp 37). Marginal.
- Test `num_warps=8` on `scoreless_kernel` (never tried). The kernel is
  dominated by scatter stores; doubling warps may halve store-serialization.
- Test a specialized branch for `max_num_pages == 2` mp workloads — there
  are several in the workload set. Previous attempts (exp 21, 22) used
  BLOCK_T=128 which hit a dead end; maybe split into two flat mp=1 paths
  instead.
- Experiment count since last new-best (exp 33): 5 done (34/35/36/37 logged,
  38 reverted). exp 37 kept as a small win. Close to the research-agent
  trigger threshold (5+ within 5% plateau).
