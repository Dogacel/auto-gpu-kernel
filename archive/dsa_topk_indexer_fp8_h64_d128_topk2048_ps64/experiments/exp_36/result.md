---
exp: 36
date: 2026-04-17
status: reverted
parent: exp_33
---

# Experiment 36 — 2026-04-17

**Description:** Skip iter 0 of `radix_topk_kernel`'s 32-iteration bit loop.
Hypothesis (INCORRECT): all real scores are non-negative (mono ≥ 0x80000000),
so iter 0's `count(mono >= 0x80000000) ≥ topk` always holds on the scoring
branch, making iter 0 a deterministic no-op. Planned save: 1/32 tree
reductions ≈ ~0.5 µs per slow-path workload.

## Implementation

Two attempts, both failed correctness:

**Variant A** (skip iter 0, init threshold to 0x80000000):
```python
threshold = tl.full([BLOCK_N], 0x80000000, tl.uint32)
for i in tl.static_range(1, 32):
    ...
```

**Variant B** (init threshold to 0x80000000, keep full loop):
```python
threshold = tl.full([BLOCK_N], 0x80000000, tl.uint32)
for i in tl.static_range(0, 32):
    ...
```

## Results

- Pass: 8/16 (all fast-path pass; ALL slow-path fail with "out-of-range indices")
- Specific failures (variant B, stride 8):
  - 4c7705ad batch [2], 19e7663d batch [5], f457feb2 batch [5], 7f1cd9c2 batch [5]
  - de54c4e6 batch [2], 2f3b7321 batch [2], e63194e7 batch [2]
  - a876010b passed (edge case: weights happen to be mostly positive in that batch)
- Mode: quick + stride 8
- **Reverted**. Restored `threshold = tl.zeros(...)` with `static_range(0, 32)`.

## Learnings

**WEIGHTS CAN BE NEGATIVE.** The score formula is:
```
final_score[b,t] = scale * sum_h(relu(q·K[h,t]) * weights[b,h])
```
`relu(...)` is always ≥ 0, but `weights[b,h]` is arbitrary fp32 (no sign
constraint). `scale` is non-negative (fp8 amax quant, per LESSONS.md).

So `final_score` can be NEGATIVE when most positive contributions (relu of
q·k) are paired with negative weights. For such batches:

- Negative score bits have `sign=1`, XOR 0xFFFFFFFF → mono < 0x80000000.
- If count(mono >= 0x80000000) < topk (i.e., fewer than 2048 non-negative
  scores in the batch), iter 0's original semantic REJECTS bit 31, leaving
  threshold=0, and subsequent iters search from 0x40000000 downward into
  the negative-score mono range.
- My optimization force-set threshold = 0x80000000, pre-judging bit 31,
  which over-filters: final_mask ends up with < topk entries, leaving
  some `topk_indices` slots un-written (garbage from caller).

**How it manifests as "out-of-range indices"**: caller doesn't fill
`topk_indices` with -1 before the dispatcher (the radix kernel guarantees
complete overwrite under the correctness invariant). With fewer-than-topk
writes, the bench sees whatever was in the buffer — often integers > total
token count → "out-of-range".

## Takeaways

1. **Verify ALL score-sign assumptions before pre-judging bits.** I assumed
   relu → scores ≥ 0 without checking that weights can flip signs after
   the per-head multiply-and-sum. The formula `sum_h(relu(...) * w[h])` is
   NOT non-negative when `w` has negative entries.

2. **The "exactly topk writes" correctness invariant is strict.** Any
   radix micro-opt that changes the pool of considered-real scores must
   preserve `count(mono >= final_threshold) ≥ topk`. Starting threshold
   higher than 0 can over-filter.

3. **If skipping iter 0 is worth it**, it needs a conditional: check
   count(mono >= 0x80000000) upfront; if ≥ topk, skip; else keep iter 0.
   That's one tl.sum of savings minus one conditional branch — ambiguous
   net win given branch overhead.

## Next candidates

- `strict_count = tl.max(strict_prefix)` (replaces one tl.sum with tl.max,
  reusing packed_prefix) — net tree-reduction budget unchanged, but fuses
  post-loop reductions. Probably no-op.
- `num_warps` on scoreless_kernel or fast_small_kernel (never tested).
- Research agent: we're at 3 consecutive reverts since exp 33 (34/35/36).
  One more miss and we should call research.
