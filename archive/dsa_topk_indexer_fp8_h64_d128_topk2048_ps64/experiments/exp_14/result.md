---
exp: 14
date: 2026-04-17
status: reverted
parent: exp_10
---

# Result — Overlap `.item()` sync with score_kernel execution (reverted)

## Change
Same dynamic `effective_topk` idea as exp 13, but moved the
`seq_lens.max().item()` sync *after* the score_kernel launch so CPU
stall overlaps with GPU execution. Kept the score allocation and grid
at their full sizes (no shrink) so the overlap-candidate workloads
still get the optimization.

```python
score_kernel[grid](...)  # launch first
if max_scored > topk:
    max_sl = int(seq_lens.max().item())
    effective_topk = min(topk, max(1, max_sl))
else:
    effective_topk = max_scored
_, topk_idx = torch.topk(scores, effective_topk, dim=-1)
```

## Measurement
A/B vs exp 10 (paired, same VM):

```
UUID           A (ms)     B (ms)    ΔB−A (ms)        %  winner
05775386       0.0488     0.0488      -0.0000   -0.03%  B
19e7663d       0.0569     0.0898      +0.0330  +57.95%  A
2f3b7321       0.0591     0.0918      +0.0327  +55.44%  A
30cecff1       0.0251     0.0248      -0.0003   -1.24%  B
4c7705ad       0.0528     0.0878      +0.0350  +66.18%  A
6caf09cf       0.0487     0.0487      -0.0000   -0.06%  B
7f1cd9c2       0.0572     0.0894      +0.0322  +56.28%  A
9c313fc4       0.0496     0.0495      -0.0000   -0.03%  B
a876010b       0.0817     0.1161      +0.0344  +42.09%  A
bb22d09a       0.0528     0.0529      +0.0000   +0.01%  A
de54c4e6       0.0590     0.0917      +0.0327  +55.41%  A
df80c00b       0.0487     0.0487      -0.0000   -0.05%  B
e49574dd       0.0394     0.0394      -0.0000   -0.01%  B
e515e20a       0.0528     0.0528      +0.0000   +0.00%  A
e63194e7       0.0590     0.0918      +0.0328  +55.54%  A
f457feb2       0.0574     0.0895      +0.0321  +55.88%  A

Paired n=16 | B wins 6/16 | mean Δ = +0.0165 ms → A faster
```

## Why it lost

The overlap pattern works in theory but **the .item() stall still
blocks CPU-side issue of the next kernel (torch.topk)**. Even if
score_kernel runs concurrently with the sync, the CPU wakes up after
the sync to issue torch.topk → a GPU pipeline gap of ~30 µs between
score_kernel and torch.topk.

Break-down of the 33 µs regression:
- Sync itself: ~20-30 µs (CUDA runtime D→H sync overhead, effectively a floor).
- Kernel-to-kernel pipeline gap: ~10 µs (CPU wakes, re-issues topk).
- Savings on torch.topk (K=2048 → K≈max_sl): 0 µs on these workloads,
  because they all have max_sl ≥ ~1500 (observed from workload_profile).

The 4 workloads that didn't regress (05775386, 30cecff1, 9c313fc4,
e49574dd, df80c00b, 6caf09cf, bb22d09a, e515e20a) all have
`max_scored <= topk` (max_num_pages ≤ 32), so they take the no-sync
path — matching exp 10 exactly.

The 8 workloads that regressed badly all have max_num_pages > 32 → sync
taken → +33 µs penalty.

## Lesson

Even when the sync is issued *after* the kernel launch, the
cross-kernel pipeline stalls. **Host-side `.item()` is a structural
barrier** between GPU ops on the default stream, not just a D→H
transfer — it prevents the CPU from queuing the next kernel until
after GPU completion of prior stream ops.

To make `seq_lens`-derived hyperparams useful, we need either:
1. A true async path (separate stream, speculative launches, or
   kernel-side consumption of the GPU value without CPU sync).
2. A way to eliminate the need for it entirely (e.g., a custom top-K
   kernel that takes K from a GPU-side scalar).

Neither is a single-iteration change.

## Reverted to exp 10 state

Same final state as before exp 13: fixed K=2048, full grid, full
scores allocation. Proceed to exp 15 on a different axis.
