---
exp: 12
date: 2026-04-17
status: reverted
parent: exp_10
---

# Result — Bump remap_kernel BLOCK_K 256 → 1024 (reverted)

## Change
Changed `BLOCK_K = 256` to `BLOCK_K = 1024` in the Python launch
prep, reducing `remap_grid` from `(B, 8)` to `(B, 2)` for
`topk=2048`.

## Measurement

A/B vs exp 10 (paired, same VM):

```
Paired n=16 | B wins 0/16 | mean Δ = +0.0004 ms → A faster
```

Systematic regression across every workload, 0.5-1.6% each. No
workload improved.

## Why it lost

Hypothesis was launch overhead dominates on this tiny op — turns
out 256 is actually the right tile for remap_kernel on B200:

- Per-program work scales linearly (1024 elts = 4× the divmod /
  table-lookup / int ops of 256 elts) while Triton's program
  dispatch on B200 is very cheap.
- Larger tile probably pushes up register pressure: int64
  topk_idx + int32 page_idx + int32 offset + int64 global_page +
  int32 token_idx = ~4 int64 worth of live state × 1024 elements.
  That's 32 KB live state (for int64) per program, or 256 regs
  per thread with num_warps=4 — brushing against the 255-per-thread
  limit. Likely some spills.
- Fewer programs also means worse SM occupancy when
  `batch_size=1` — only 2 programs vs 8 for the whole kernel.

## Reverted to exp 10 state

`BLOCK_K = 256` restored. Proceed to exp 13 on a different axis.

## Takeaway
For trivially-cheap fused-epilogue kernels (remap, mask-write,
final cast), **don't** assume bigger tile = less overhead. The
per-program work is so small that dispatch-to-work ratio is
already favorable at 256-element tiles, and 1024-element tiles
just add register pressure without improving occupancy.
