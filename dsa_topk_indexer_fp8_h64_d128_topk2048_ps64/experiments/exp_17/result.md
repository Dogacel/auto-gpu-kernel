---
exp: 17
date: 2026-04-17
status: reverted
parent: exp_10
---

# Result — Skip `torch.as_strided`, raw dtype-reinterpreted tensors (reverted)

## Change

Replaced the two `torch.as_strided` view-construction calls with
direct dtype reinterpretations on the underlying storage:

```python
k_fp8 = k_index_cache_fp8.view(torch.float8_e4m3fn)  # was as_strided
k_scale = k_index_cache_fp8.view(torch.float32)       # was as_strided
```

The kernel receives explicit strides and a new `SCALE_OFFSET` constexpr
(=2048 fp32 elements) that encodes the per-page offset from the fp8
region start to the scale region start — previously applied via
`storage_offset` in `as_strided`.

Also hardcoded spec-fixed constants (`H=64`, `D=128`, `page_size=64`,
`head_dim_sf=132`, `topk=2048`) to skip shape-unpacking for these axes.

## Results
- Pass: 2/2 quick (exact match, abs_err=rel_err=0)
- Mode: quick + ab-vs-exp_10

## Measurement

A/B vs exp 10 (paired, same VM):

```
Paired n=16 | B wins 6/16 | mean Δ = +0.0001 ms → A faster
```

Per-workload deltas are all within ±0.0004 ms (±0.7%). No systematic
pattern — about half small wins, half small regressions; all below
noise floor.

## Why it lost

The `torch.as_strided` call overhead is smaller than estimated. It
probably costs ~0.5 µs per call (not 1.5-2 µs), and both calls together
are ~1 µs — within the same noise band as the A/B mean Δ of +0.1 µs.

The profile phase labeled `py_setup` (12.5 µs) apparently doesn't have
`as_strided` as its dominant component; more likely it's the
`torch.empty` dispatch (9.4-9.8 µs, exp 16 covered this and lost) plus
the kernel launch path's own setup. Removing `as_strided` shaves
essentially nothing measurable.

## Lesson

`torch.as_strided` on the hot path is cheaper than it looks (~0.5 µs
per call, not a few µs). Eliminating it doesn't recover meaningful
latency. Along with exp 16 (module-level scores cache = tied) and
exp 13/14 (`.item()` sync = structural barrier), this confirms the
py_setup + alloc + small-dispatch phase has no easy single-µs level
optimization left. **The 12.5 µs `py_setup` measurement is dominated
by torch dispatch machinery we can't trim from Python.**

## Reverted to exp 10 state

New code discarded. Proceed to exp 18 on a different axis — likely
score_kernel tile tuning (BLOCK_T with 2 pages per program via
non-looping tile merge), or Gluon migration if that fails too.
