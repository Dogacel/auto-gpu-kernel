---
exp: 24
date: 2026-04-17
status: reverted
parent: exp_20
---

# Experiment 24 — 2026-04-17

**Description:** Added `num_stages=3` kwarg to the `score_kernel[grid](...)` launch
in the non-fast-path dispatch. Goal: increase prefetch depth so Triton overlaps the
q/k/scale/w loads more aggressively behind the single fp8 `tl.dot`. Default was
unset, which typically resolves to `num_stages=1` or similar.

## Results
- Pass: **16/16** (quick 2/2 + stride 8)
- Kernel latency (ms, stride 8): min=0.011 / mean≈0.046 / max=0.074
- A/B vs exp 20 (paired, stride 8): **B wins 9/16, mean Δ = +0.0000 ms (tied)**
- Mode: quick (passed) + stride 8 + ab-vs-exp_20

## A/B per-workload breakdown

All 16 workloads within ±1.5% (typical cross-run noise). No workload
shows a material delta either direction. Two slightly larger absolute
changes:

| uuid       | A (ms) | B (ms) | Δ (ms)   | %      |
|------------|-------:|-------:|---------:|-------:|
| e49574dd   | 0.0414 | 0.0408 |  -0.0006 |  -1.51% |
| 30cecff1   | 0.0086 | 0.0085 |  -0.0001 |  -1.33% |
| 9c313fc4   | 0.0490 | 0.0494 |  +0.0004 |  +0.84% |
| df80c00b   | 0.0489 | 0.0493 |  +0.0004 |  +0.87% |

All within noise.

## Why it's a no-op

`score_kernel` has no outer loop over tile indices — it performs a single
`tl.dot` on pre-loaded q/k tiles. Triton's `num_stages` attribute pipelines
iterations of an outer loop by staggering global-memory loads with compute.
With zero iterations to overlap, the attribute has nothing to schedule.

The q/k loads are already issued before the dot, and the dot itself launches
an async MMA on Blackwell tensor cores — single-stage is effectively already
an async pipeline. No more prefetch slots to fill.

## Lessons

- **`num_stages` is a no-op on kernels without outer loops.** The attribute
  pipelines load/compute across loop iterations. For a straight-line kernel
  (single dot, no iteration), it has no effect on codegen.
- Corollary: tuning `num_stages` is only worth trying on kernels with
  `for` loops over K tiles or similar iteration. Both `score_kernel` and
  `fast_small_kernel` here are single-iteration, so this lever is exhausted.

## Reverted to exp 20

## Next candidate

We're 4 reverts deep since exp 20 (exp 21, 22, 23, 24). Pattern:
- Tile-tuning levers (num_warps, num_stages, BLOCK_K, BLOCK_T=128, 2-dot) all
  tied or regressed.
- Structural levers tried: mp=2 fast path (3 variants, all regressed),
  alloc elimination (cache + alias, both tied or regressed).
- Biggest remaining bottleneck per `profile.md`: `torch.topk` at 49-61 µs on
  medium/large workloads. Replacement requires a multi-iteration custom
  top-K rewrite.

Per CLAUDE.md loop guidance: consider spawning research agent for a fresh
plan, or approaching Gluon migration threshold (after 15-20 stuck iterations).
Current stuck streak: 4 consecutive reverts. Not yet at Gluon threshold but
research-agent trigger is justified (≥5 experiments within 5% of each other
from exp 20 onwards).
