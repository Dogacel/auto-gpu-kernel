---
exp: 19
date: 2026-04-17
status: reverted
parent: exp_10
---

# Result — BLOCK_T=128 via two independent dots with store-between (reverted, regression)

## Change

Alternative implementation of two-pages-per-program: instead of
concatenating K tiles into [128, 128] and doing one big MMA (exp 18),
do TWO independent [64, 128] @ [128, 64] dots with `tl.store` between
them to serialize register usage:

```python
# Shared: load Q, w once per program
# Page 0: load k0, scale_0 → dot → relu*w → sum*scale → store (freed)
# Page 1: load k1, scale_1 → dot → relu*w → sum*scale → store
```

Grid halved the same way as exp 18. `max_num_pages` passed for the
clamp. Hypothesis was that avoiding the `tl.trans`+`tl.reshape`
overhead would fix exp 18's medium regressions.

## Results
- Pass: 2/2 quick (exact match)
- Mode: quick + ab-vs-exp_10

## Measurement

A/B vs exp 10:

```
Paired n=16 | B wins 1/16 | mean Δ = +0.0024 ms → A faster
```

This is **much worse** than exp 18 (+0.5 µs). Only the largest
workload (a876010b, -3.68%) wins; all other 15 regress by +1 to +8%.

| uuid | Δ % |
|---|---:|
| 4c7705ad | +8.34% |
| 05775386 | +8.01% |
| e49574dd | +7.48% |
| f457feb2 | +7.04% |
| 6caf09cf | +6.83% |
| e515e20a | +6.38% |
| df80c00b | +6.49% |
| 9c313fc4 | +6.20% |
| 19e7663d | +6.15% |
| 30cecff1 | +6.01% |
| bb22d09a | +4.74% |
| 7f1cd9c2 | +4.57% |
| 2f3b7321 | +1.76% |
| de54c4e6 | +1.17% |
| e63194e7 | +1.09% |
| **a876010b** | **-3.68%** |

Even the smallest workload (30cecff1), which benefited +16% from
exp 18, now regresses +6%.

## Why it lost (and what exp 18 actually told us)

The two-dot structure is exactly what `LESSONS.md` warns about under
"Grid shape": `PAGES_PER_PROGRAM > 1 (loop multiple pages per program)
regressed big`. Triton appears to fuse the two dots into a
compound pattern that loses the per-program pipelining benefit. The
`tl.store` between them is visible to the user but the compiler still
schedules both as a unit.

**Exp 18's big-MMA approach was actually the BETTER implementation
of BLOCK_T=128.** The `tl.trans`+`tl.reshape` cost was real but
smaller than the pipelining loss we see with two-dot. Exp 18's
medium regression is inherent to halved-grid pipelining loss, NOT
layout conversion.

## Lesson

On this kernel, two separate dots per program is strictly worse than
either (a) one per program (exp 10, current best), or (b) one bigger
MMA via tile concatenation (exp 18, mixed but less bad than exp 19).
The fused big-MMA path at least gets WGMMA m64n128 throughput; the
two-dot path gets neither grid-halving benefit nor larger-tile
throughput — only the launch-overhead cost of fewer programs without
any computational win.

## Reverted to exp 10

Next candidate: **exp 20 = exp 18 (big MMA) + num_warps=8**, to test
the plan's listed risk-mitigation for register pressure. If that also
fails, BLOCK_T=128 is dead and we should pivot (possibly to Gluon
after 2-3 more failed iterations on other axes).
