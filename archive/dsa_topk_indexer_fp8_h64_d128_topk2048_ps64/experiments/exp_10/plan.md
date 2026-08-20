# Experiment 10 — Apply k_scale after sum reduction (skip per-element multiply)

## Goal

Reorder the score_kernel so that `k_scale` is applied as a scalar-per-token
multiply **after** the cross-head sum, not as a 64×64 broadcast multiply
before relu.

Savings: one 64×64 fp32 elementwise multiply per active program.

## Correctness

Current code computes:
```
scores[h,t] = q_dot[h,t] * scale[t]        # broadcast mul
scores[h,t] = max(scores[h,t], 0)          # relu
scores[h,t] = scores[h,t] * w[h]           # broadcast mul
final[t]   = sum_h(scores[h,t])
```

Proposed:
```
scores[h,t] = max(q_dot[h,t], 0)           # relu (scale-free)
scores[h,t] = scores[h,t] * w[h]
partial[t]  = sum_h(scores[h,t])
final[t]    = partial[t] * scale[t]        # scalar per t
```

Invariant: `scale[t] >= 0` (deep_gemm quant stores `amax/fp8_max`).
For non-negative scale, `max(x * s, 0) == s * max(x, 0)`, and sum is
linear: `sum_h(s * max(x_h, 0) * w_h) = s * sum_h(max(x_h, 0) * w_h)`.

## Implementation

Move the scale load and multiply out of the H-broadcast region:
- Keep `scale = tl.load(...)` (same place or slightly earlier to
  overlap with matmul)
- Drop `scores = scores * scale[None, :]` before relu
- After `final = tl.sum(scores, axis=0)`, insert `final = final * scale`

Also hoist `w` load earlier (before `tl.dot`) so both scalar loads can
overlap with matmul latency. Neutral for correctness.

## Expected win

Small: saves 64×64 = 4096 fp32 muls per program, ~500K fp32 muls
per large workload. On B200 that's ~20 ns compute saved per program
(not counting register pressure effects). Net target: −1% to −3% on
stride 8, or neutral.

## Risks

- **Scale sign assumption**: if deep_gemm ever emits negative scales,
  relu-commute breaks. Based on LESSONS.md and baseline code, the
  scale is a positive amax ratio — safe.
- Load-reorder could increase register pressure by holding `scale`
  and `w` across the matmul; if that regresses, revert the hoist
  while keeping the algebraic reorder.

## Success criterion

- A/B vs exp 9: mean Δ ≤ 0 (not a loss); a small improvement is
  fine — correctness is the bar.
- Correctness: 128/128 pass, exact element match.
