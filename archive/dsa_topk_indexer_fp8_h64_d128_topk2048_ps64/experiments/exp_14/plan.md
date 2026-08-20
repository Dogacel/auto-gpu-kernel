# Experiment 14 — Overlap `.item()` sync with score_kernel execution

## Goal

Reduce `torch.topk`'s K from fixed 2048 to `min(2048, max(seq_lens))` —
same optimization as exp 13 — but without the 60 µs serial stall exp 13
paid. The insight: **launch `score_kernel` first**, *then* call
`.item()` while the GPU is busy on the kernel, *then* call
`torch.topk` with the dynamic K.

## Workload evidence

- 69/128 workloads have all `seq_len < 2048` (padded topk sort).
- 27 workloads could use K ≤ 1024; 13 could use K ≤ 128.
- Mean early_return_frac = 70.9% → score_kernel finishes the active
  portion quickly, but the kernel's *wall-clock* time (including
  early-return programs) is ~30-40 µs on large workloads.
- On small workloads, score_kernel wall-clock is ~5-15 µs and may not
  fully hide the sync. That's ok — on small workloads we skip the sync
  path entirely via a threshold.

## Approach

```python
max_scored = max_num_pages * page_size
scores = torch.empty((batch_size, max_scored), ..., dtype=torch.float32)

grid = (batch_size, max_num_pages)  # full grid — DON'T shrink
score_kernel[grid](..., scores, ...)  # launch first, kernel starts on GPU

# Only pay the sync when there's meaningful headroom to shrink topk K.
if max_scored > topk:
    # GPU is now busy with score_kernel. `.item()` stalls CPU until
    # seq_lens.max() is ready — but the kernel execution overlaps with
    # the sync, so net cost is ~kernel_time or ~sync_time, whichever
    # is larger — not their sum.
    max_sl = int(seq_lens.max().item())
    effective_topk = min(topk, max(1, max_sl))
else:
    effective_topk = max_scored

_, topk_idx = torch.topk(scores, effective_topk, dim=-1)
```

Kernel internals and score allocation size are **unchanged** from exp 10.
We do **not** shrink the score_kernel grid or scores allocation — that
was exp 13's second axis and it saved <5 µs, not worth the complexity.
Only the torch.topk K is dynamic.

## Why this should work when exp 13 didn't

| Axis | Exp 13 | Exp 14 |
|---|---|---|
| Sync timing | *Before* kernel launch | *After* kernel launch |
| Sync observable cost | ~60 µs serial | overlaps with kernel (~30-40 µs on large) |
| torch.topk K savings | ~5-15 µs small-K workloads | same |
| Grid shrink savings | ~2-5 µs | not pursued |
| Scores alloc savings | marginal | not pursued |
| Net expected on small | +60 − 15 = +45 µs (regression) | max(kernel, sync) − full_topk ≈ −5 to −10 µs |
| Net expected on large | already 2048 K, sync skipped | same, sync skipped |

The critical insight from exp 13: **`.item()` is not a cheap D→H
transfer**, it's a CPU stall that blocks any subsequent CUDA dispatch
on the default stream. To make the stall "free" we must launch enough
GPU work before it that the CPU-side wait is hidden by GPU execution.

## Risks

- **Small workloads where score_kernel finishes faster than the sync
  latency**: on a batch=1 workload with sl=64, score_kernel is ~5 µs
  but sync may still be ~20 µs → net penalty.
  - Mitigation: unclear how many small workloads fall in this regime;
    need to measure. If A/B shows regressions on the smallest
    workloads, add a second threshold (e.g. skip sync when
    `batch_size * max_num_pages < K`).
- **CUDA stream ordering**: `torch.topk` and `.item()` on
  `seq_lens.max()` are on the default stream. `.item()` synchronizes
  to the host; `torch.topk` is queued on the GPU. The sync only waits
  for `seq_lens.max()` (which doesn't depend on score_kernel output),
  so in principle it could return *before* score_kernel finishes — in
  which case we haven't overlapped anything.
  - **Critical question**: does `seq_lens.max()` queue behind
    score_kernel on the default stream, or is it allowed to complete
    sooner? Since both ops share the default stream and are issued
    in-order, `seq_lens.max()` should enqueue *after* score_kernel.
    The sync then waits for *everything ahead* to complete, including
    score_kernel.

## Success criterion

- Full mean Δ ≤ −3% vs exp 10 (0.047 ms → 0.0456 ms).
- A/B vs exp 10: at least 10/16 wins, mean Δ ≤ −1 µs.
- Correctness: 128/128 exact match.

## Fallback if it regresses

If A/B still regresses, try a simpler variant: compute
`effective_topk` *entirely on GPU* and pass to torch.topk without a
sync. This isn't possible with stock `torch.topk` (requires Python
int), but we could push the scores+K into a custom top-K kernel — out
of scope for a tile-tuning iteration.
