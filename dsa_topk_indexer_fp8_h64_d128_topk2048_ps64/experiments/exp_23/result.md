---
exp: 23
date: 2026-04-17
status: reverted
parent: exp_20
---

# Result — Alias DPS output as fp32 scratch (reverted, regression)

## Change

Per `exp_23/plan.md`, replaced:

```python
scores = torch.empty((batch_size, max_scored), dtype=torch.float32)
```

with a DPS-aliased view when `max_scored <= topk (2048)`:

```python
if max_scored <= topk:
    scores = topk_indices.view(torch.float32)[:, :max_scored]
else:
    scores = torch.empty((batch_size, max_scored), dtype=torch.float32)
```

Goal: skip the ~9 µs `torch.empty` dispatch measured in `profile.md`.
Ceiling estimated ~4.3 µs mean across the 61/128 workloads with
`max_scored ≤ 2048`.

## Results
- Pass: **16/16** (stride 8 quick + A/B)
- Kernel latency (ms, stride 8): min=0.0090 / mean=0.0471 / median=0.0480 / max=0.0750
- A/B vs exp 20 (paired, stride 8): **B wins 8/16, mean Δ = +0.0016 ms → A faster (exp 20 is better)**
- Mode: quick (passed) + stride 8 + ab-vs-exp_20

## A/B per-workload breakdown

5 workloads regressed by +9-13% (~+5 µs each):

| uuid       | A (ms) | B (ms) | Δ (ms)  | %      |
|------------|-------:|-------:|--------:|-------:|
| 05775386   | 0.0489 | 0.0538 | +0.0049 | +10.1% |
| 6caf09cf   | 0.0489 | 0.0537 | +0.0048 |  +9.7% |
| 9c313fc4   | 0.0492 | 0.0538 | +0.0046 |  +9.4% |
| df80c00b   | 0.0489 | 0.0537 | +0.0049 | +10.0% |
| e49574dd   | 0.0414 | 0.0469 | +0.0055 | +13.2% |

11 workloads effectively tied (|Δ| ≤ 0.7 µs each); none materially
improved. The expected 9 µs savings did not materialize on any
workload — no Δ ≤ -1 µs observed.

## Why it lost

The ceiling analysis anticipated savings on all mp∈[2,32] workloads.
Observed: those workloads **regressed** by ~5 µs each. The
hypothesized failure mode (d) from the plan ("torch.empty was actually
cheaper than profile.md suggested") is partially right, but the
additional finding is:

**torch.topk has a fast path for contiguous input.** The aliased view
`topk_indices.view(torch.float32)[:, :max_scored]` has shape
`(B, max_scored)` but stride `(2048, 1)` — non-contiguous along the
batch dim when `max_scored < 2048`. `torch.empty((B, max_scored))`
has stride `(max_scored, 1)` — contiguous. The strided aliased input
forces a slower `torch.topk` kernel, adding ~5 µs per call and
wiping out the ~9 µs saving.

This is a new lesson distinct from exp 16's finding (that a Python
dict cache doesn't help). Exp 16 suggested: "you need to eliminate
the `torch.empty` call itself (e.g., pass in a pre-allocated buffer
as an argument)". This experiment tested eliminating the call via
aliasing — and revealed that the *downstream consumer* has a layout
dependence that costs as much as the alloc itself.

## Lessons

1. **DPS aliasing with slicing creates strided views.** When the
   source tensor is wider than needed (2048 int32) and the aliased
   region is narrower (max_scored fp32 < 2048), the view is strided
   along the batch dim. This breaks `torch.topk`'s contiguous-input
   fast path.
2. **Eliminating `torch.empty` is not a free saving.** The ~9 µs
   dispatch cost hides a second layout benefit: the fresh contiguous
   tensor. Together they're ~9 µs alloc + ~5 µs fast-path unlock.
   Replacing just the alloc loses the layout benefit.

## Reverted to exp 20

## Next candidate

Per `exp_23/plan.md` follow-up section: **Opt #2 (flat-grid score_kernel)**
from `workload_profile.md`. The current `(B, max_num_pages)` grid wastes
70.9% of programs (early-returned via `token_start >= seq_len`). A flat
`(sum_active_programs,)` grid with indirect addressing via a prefix-sum
lookup could reclaim those slots. Ceiling: 1-3 µs mean × 128 workloads.

Alternative: try the alloc-elimination idea **without** slicing — alias
the full `topk_indices.view(torch.float32)` as a [B, 2048] buffer, padding
the unused columns to -inf so `torch.topk` still picks the right top-K
from the full 2048. Cost: add a `fill_` call (~1-3 µs); benefit: preserves
contiguous layout. This may or may not clear the torch.topk fast path
check — needs testing.
