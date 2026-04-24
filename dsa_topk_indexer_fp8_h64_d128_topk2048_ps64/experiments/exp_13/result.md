---
exp: 13
date: 2026-04-17
status: reverted
parent: exp_10
---

# Result — Dynamic effective_topk + shrunk grid via `seq_lens.max().item()` (reverted)

## Change
Added `.item()` sync on `seq_lens.max()` *before* `score_kernel` launch,
then shrunk `grid_p`, `scores` allocation width, and `effective_topk` to
the active range.

## Measurement
A/B vs exp 10 (paired, same VM):

```
Paired n=16 | B wins 3/16 | mean Δ = +0.0294 ms → A faster
```

Massive regression: most workloads doubled in absolute latency
(e.g. 0.057 → 0.117 ms, +120% on some). Only 3 of 16 workloads won.

## Why it lost

The `.item()` sync costs ~60 µs *in practice*, not the ~5 µs I'd
estimated from raw wire latency. On the benchmark harness the sync is
serial with kernel launch — the Python path stalls on
`int(seq_lens.max().item())` before `score_kernel[grid](...)` dispatches.
Any savings on `torch.topk` (≤15 µs on best case) or grid shrink
(≤5 µs) can't cover a 60 µs stall.

## Lesson

Sync *before* a kernel launch that could otherwise execute is far more
expensive than the raw device→host transfer implies — it also blocks
kernel dispatch. To exploit `seq_lens`-derived hyperparams, the sync
must overlap with kernel work (launch first, sync while GPU is busy,
then adjust downstream ops).

## Reverted to exp 10 state

All dynamic sizing removed. Proceed to exp 14 which tries the overlap
pattern: launch score_kernel with full grid first, *then* call
`.item()` while kernel runs, then slice `scores` and call `torch.topk`
with the dynamic K.
