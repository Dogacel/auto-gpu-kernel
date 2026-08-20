---
exp: 31
date: 2026-04-17
status: reverted
parent: exp_29
---

# Experiment 31 — 2026-04-17

**Description:** Module-level pool for the `scores` buffer (simple single-tensor
pool with grow-only semantics, no dict keyed by shape). Goal: skip the ~5-10 µs
`torch.empty` dispatch per call on slow-path workloads.

## Implementation

```python
_SCORES_POOL = None
_SCORES_POOL_CAP = 0

@torch.no_grad()
def kernel(...):
    global _SCORES_POOL, _SCORES_POOL_CAP
    ...
    need = batch_size * max_scored
    if need > _SCORES_POOL_CAP:
        _SCORES_POOL = torch.empty(max(need, 512 * 1024), device=device, dtype=torch.float32)
        _SCORES_POOL_CAP = _SCORES_POOL.numel()
    scores = _SCORES_POOL[:need].view(batch_size, max_scored)
```

Avoids exp 16's dict lookup (0.3 µs tuple-key). Allocation happens once per
process (after first slow-path workload bumps the cap to a typical size).

## Results

- Pass: 2/2 quick
- A/B vs exp 29 (stride 8, paired, same VM):
  - B wins 9/16, mean Δ = **-0.0000 ms** → B faster (tied within noise)
  - Slow-path: a876010b +0.08% (noise), 2f3b7321 -0.14%, 4c7705ad -0.56% — no meaningful shift
- Mode: quick + ab-vs-exp_29
- **Reverted.**

## Learnings

1. **Reconfirmation of exp 16 lesson post-radix.** After removing `torch.topk`
   (exp 28) the "alloc" phase didn't become relatively more expensive — it's
   still ~5-10 µs of mostly Python-side tensor object creation + caching-allocator
   lookup, and a Python-level pool doesn't escape that. The returned tensor is
   still a new Python object (`_SCORES_POOL[:need].view(...)` creates two new
   views, each ~1 µs).
2. **The only remaining lever for alloc elimination** is to skip the PyTorch
   tensor object entirely — e.g., pass a raw pointer + shape to the kernel as
   Python ints. Triton's JIT launcher accepts pointers; we could cache the pool
   as a pointer and pass it directly. Not a single-iteration change, and the
   upside is ≤5 µs.
3. **`torch.empty` dispatch cost is a floor, not a bottleneck.** Before spending
   more iterations on it, need profile data confirming its share of current
   slow-path latency has grown post-radix. Exp 10 profile (9.8 µs) is probably
   still accurate in absolute µs, but slow-path total dropped from 123 µs to
   ~35 µs, so alloc is now ~28% of slow-path if unchanged. That's a bigger
   relative share — worth revisiting if a zero-alloc path becomes available.

## Next candidates

- Call profiler agent — structural changes (exp 28/29) mean exp 10 profile is
  obsolete; need a current breakdown before picking further levers.
- Compile-time specialization: separate `radix_topk_4k` and `radix_topk_8k`
  kernel entries so each gets its own autotune config (num_warps, num_stages).
- Fuse score + radix when `batch_size` is small enough that serial page loop
  fits in one kernel (e.g., B ≤ 8 and mp ≤ 16): save scores HBM roundtrip.
- Shrink the radix bit loop: after iteration 23 or so, the low bits rarely flip
  the set — could use a dynamic escape condition if Triton supports it.
