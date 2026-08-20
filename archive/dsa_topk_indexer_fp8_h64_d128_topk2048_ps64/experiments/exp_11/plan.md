# Experiment 11 — Re-try num_warps=8 on score_kernel

## Goal

Set `num_warps=8` on the score_kernel launch to see if the altered
kernel body (post-exp-9 early-return, post-exp-10 scale reorder)
benefits from more warps.

## Prior

Exp 5 tried `num_warps=8` on the **pre-fused** kernel (before
exp 9/10): 11/16 wins, mean Δ −0.17%, reverted as noise.

Since then the kernel body has shrunk:
- Early-return short-circuits 94% of programs (just a sentinel
  store) — more warps = faster store.
- Scale multiply moved from 64×64 broadcast to scalar-per-t —
  slightly lower register pressure per warp.

Could tip the balance.

## Change

Single-line at the `score_kernel[grid](...)` call:
```python
num_warps=8,
```

If `num_warps=8` regresses, also try `num_warps=2` (the kernel is
quite small post-exp-10).

## Success criterion

- Stride 8 mean Δ ≤ −2% vs exp 10 → confirm with A/B, keep.
- Stride 8 mean Δ between −2% and +2% → drop.
- Regression: revert, try different axis.

## Risk

Minor — single parameter change, trivially reversible.
