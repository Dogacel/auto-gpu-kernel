# Experiment 12 — Bump remap_kernel BLOCK_K to reduce launch count

## Goal

Reduce the program count of `remap_kernel` so each batch does 1–2
programs instead of 8 — cuts launch overhead on a trivially-cheap
kernel.

## Current state (exp 10)

```python
BLOCK_K = 256
remap_grid = (batch_size, triton.cdiv(topk, BLOCK_K))  # (B, 8)
```

For `topk=2048`, that's 8 programs per batch. Each program does
256 trivial int ops (divmod, table lookup, mul-add, store).
Post-exp-7 profile had remap_kernel ≈ 7 µs, ~15% of total.

## Hypothesis

A 256-int64 / 256-int32 tile is tiny for a B200 program — most of
the 7 µs is launch + grid scheduling, not the inner work. Larger
`BLOCK_K` means fewer programs, less launch overhead, same data
volume.

## Proposed change

Bump `BLOCK_K` from 256 → **1024** (4× fewer programs):
- `remap_grid = (B, 2)` for topk=2048
- 1024 int64 loads per program (8 KB) — well within register/shmem
  budget on B200
- All bounds-masking already in place (`k_in_topk`, `k_in_effective`,
  `k_in_range`) handles the pad-to-topk edge cleanly

If 1024 compiles fine and wins, try 2048 (1 program per batch) too.

## Risk
- Register pressure: 1024 int64 = 2048 int32 registers per program.
  With num_warps=4 × 32 threads = 128 threads, that's 16 regs each.
  Plus temporaries for page_idx, offset, global_page. Tight but
  should fit — default 255-reg budget per thread.
- If the compiler spills to local memory, we'd regress — watch for
  that.

## Success criterion

- Stride 8 mean Δ ≤ −3% vs exp 10 → A/B confirm, keep.
- Marginal (−3% to 0): A/B, keep only if mean/p50 both improve.
- Regression: revert, try BLOCK_K=512 as a fallback point.
- Correctness: 128/128 exact match.
