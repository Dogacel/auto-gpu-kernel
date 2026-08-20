---
exp: 40
date: 2026-04-17
status: reverted
parent: exp_37
---

# Experiment 40 — 2026-04-17

**Description:** Add `num_warps=8` to `scoreless_kernel` launch. The
scoreless path handles 69/128 workloads (54%), averaging ~2 µs amort.
Store-heavy kernel (256 int32 writes per program) — hypothesized warp
doubling would improve HBM coalescing.

## Implementation

One-line change in wrapper:
```python
scoreless_kernel[...](...,  num_warps=8)
```

## Results

- Pass: skipped quick; both quick workloads are scoreless fast-path, already at noise floor.
- A/B vs exp_37 (stride 8): **B wins 8/16, mean Δ = +0.0000 ms (tied)**
- Fast-path detail: mixed, noise-dominated (±1-2% per workload).
- Reverted.

## Learnings

- `scoreless_kernel` at BLOCK_K=256 writes 256 int32 = 1 KB per program.
  4 warps × 64 elements = 256 B/warp = one aligned sector. Doubling
  warps halves that to 128 B/warp — doesn't reduce cache-line count.
- Confirms the general pattern (exp 5/11/24/32): `num_warps` only helps
  reduction-heavy kernels (like exp 29 radix). Store-only or MMA-only
  kernels don't benefit.

## Takeaways

1. Close the `num_warps=8 on scoreless_kernel` axis. Scoreless path is
   at noise floor — any further micro-tune needs a different mechanism
   (BLOCK_K size, grid shape, or the path itself).

## Next candidates

- Re-call research agent with the explicit framing "fusion (exp 39) lost
  massively, radix micro-tuning (exps 33/34/35/38) is plateaued, small-
  workload micro-tuning (exp 40) tied. What structural lever remains?"
- Consider warp-specialized fusion (8 warps each scoring mp/8 pages,
  then cooperating on radix) — structurally different from exp 39
  because page-level parallelism is preserved within the program.
  Risk: Triton's warp-specialization primitives may not be stable enough.
- Gluon migration. We're at 7 post-exp-33 attempts with only 1 marginal
  win; approaching the 15-20 iteration "consider Gluon" threshold.

## Experiment accounting

- Since exp 33 last-best: 34/35/36/38/39/40 reverted, 37 kept (marginal).
  7 attempts, 1 marginal win, 6 reverts including 1 major regression.
