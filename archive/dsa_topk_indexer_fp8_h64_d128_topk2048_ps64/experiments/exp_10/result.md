---
exp: 10
date: 2026-04-17
status: kept
parent: exp_9
---

# Result — Apply k_scale after cross-head sum

## Change
Moved the `scale` multiply from before relu (64×64 fp32 broadcast)
to after the cross-head `tl.sum` (64-element scalar-per-t multiply).
Also hoisted the `scale` and `w` loads above the `tl.dot` so they
can overlap with matmul latency.

```python
# Issue the scalar loads early so they overlap with matmul latency.
s_off = page_id * stride_ksp + t_offs * stride_kst
scale = tl.load(k_scale_ptr + s_off)
w_off = pid_b * stride_wb + h_offs * stride_wh
w = tl.load(w_ptr + w_off)

scores = tl.dot(q_fp8, tl.trans(k_fp8), out_dtype=tl.float32)

scores = tl.maximum(scores, 0.0)
scores = scores * w[:, None]

final = tl.sum(scores, axis=0) * scale
```

## Correctness (why the reorder is equivalent)
Deep_gemm quant emits `scale[t] = amax/fp8_max >= 0`. For
non-negative scale:
- `max(x * s, 0) == s * max(x, 0)` (relu commutes with positive scale)
- `sum_h(s[t] * max(q_dot[h,t], 0) * w[h]) == s[t] * sum_h(max(q_dot[h,t], 0) * w[h])` (linearity)

So the pre-relu broadcast multiply collapses into one scalar-per-t
multiply after the sum. Empirically: 128/128 exact match.

## Measurement

- `/benchmark full`: mean **0.0470 ms**, min 0.021, max 0.075, 128/128 pass
  - small n=24: mean 0.0259
  - medium n=84: mean 0.0474
  - large n=20: mean 0.0706
- A/B vs exp 9: B wins **15/16**, mean Δ = −0.0010 ms (−2%).
  Biggest wins on medium workloads: 9c313fc4 (−3.81%), 6caf09cf
  (−3.72%), df80c00b (−3.72%), 30cecff1 (−3.66%). Large workloads
  barely moved: a876010b (−0.31%), de54c4e6 (−0.12%) — they're
  top-K bound, not score-bound.

## Vs exp 9 (0.0478 ms mean, 0.022/0.076 min/max)
- Mean: −1.7% (full), −2% (A/B).
- Max: 0.076 → 0.075.
- Pass: 128/128, exact match.

## Why this works (modest)
Saving a 64×64 fp32 multiply per active program is tiny in raw
FLOPs, but:
- Moves the scale/weight loads off the critical path (overlap
  with matmul)
- Frees registers during the sum reduction (no broadcast multiply
  result held)

The win concentrates on medium-size workloads where score_kernel
is a meaningful fraction of total time. Large workloads are
torch.topk bound (~45% of total per profile.md), so score_kernel
savings don't translate.

## Takeaway
On Triton kernels with a downstream scalar-valued reduction, check
if the per-element multiplies commute through relu/sum and can be
folded into a single scalar multiply at the end. Saves both a
broadcast multiply and lets the scalar load overlap with the heavy
matmul.
