---
exp: 16
date: 2026-04-17
status: reverted
parent: exp_10
---

# Result — Cache scores buffer via module-level pool (reverted)

## Change
Added `_scores_cache: dict = {}` keyed by `(B, max_scored, device)`.
The kernel function checks the cache; on hit, reuse the buffer; on
miss, alloc and insert. Correctness relies on `score_kernel` writing
every position (active + early-return paths).

## Measurement
A/B vs exp 10 (paired, same VM):

```
Paired n=16 | B wins 8/16 | mean Δ = +0.0001 ms → A faster
```

Mix of tiny wins (≤0.3 µs) and tiny regressions (≤1.5 µs). The two
biggest regressions (9c313fc4 +1.3%, e515e20a +1.5%) are on medium-B
workloads where Python cache lookup adds overhead.

## Why it lost

PyTorch's CUDA caching allocator already pools freed buffers by size.
When `torch.empty((B, max_scored), ...)` is called with a shape
matching a recently-freed buffer, it returns the cached chunk
immediately. Our module-level dict cache adds a Python-level layer
that:
- Saves at best ~2-3 µs of Python dispatch on each call.
- Adds ~0.2-0.5 µs of tuple construction + dict get per call.
- Holds onto buffers across calls, which could actually slow things
  down by preventing PyTorch's allocator from reclaiming them for
  other uses.

The 9.4-9.8 µs "alloc" phase measured in the profile is NOT pure
torch.empty overhead — it includes PyTorch dispatch machinery that
can't be bypassed at the Python level.

## Lesson

Re-allocation of same-shape tensors via `torch.empty` is already well
pooled by PyTorch's caching allocator. Attempting a Python-level cache
on top of it saves little and can add tuple/dict overhead that
cancels the gain. **If you want to eliminate alloc overhead, you need
to eliminate the `torch.empty` call itself** — e.g., by using a
preallocated tensor passed in as an argument (not possible here,
since `scores` is an internal intermediate). Don't try to out-cache
PyTorch's allocator.

## Reverted to exp 10 state

`_scores_cache` removed. Proceed to exp 17 on another axis.
