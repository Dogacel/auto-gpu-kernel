---
exp: 29
date: 2026-04-17
status: new best
parent: exp_28
---

# Experiment 29 — 2026-04-17

**Description:** Add `num_warps=8` to the `radix_topk_kernel` launch. Goal: fix
the exp 28 regression on a876010b (mp=91, BLOCK_N=8192, +7%) by speeding up the
32-iteration `tl.sum` tree reductions that were single-warp-bottlenecked at
default `num_warps=4`.

## Implementation

Single-line change in `kernel()` dispatch:
```python
radix_topk_kernel[(batch_size,)](
    ...,
    BLOCK_N=BLOCK_N,
    num_warps=8,
)
```
All other code unchanged.

## Results

- Pass: **128/128**
- Kernel latency (ms) — full run: small=0.0020 / large=0.0228 / overall=**0.0116** mean (min 0.002 / max 0.035)
- Max abs err: 0.00 | Max rel err: 0.00 (matched_ratio = 1.0000)
- Mode: full + ab-vs-exp_28

### A/B vs exp 28 (stride 8, paired)

- B wins 11/16, mean Δ = **-0.0037 ms**
- **a876010b -58%** (0.0874 → 0.0367 ms) — exp 28's only regression is now the biggest win
- 6 other slow-path workloads: -5% to -7% additional gain (incremental)
- Fast-path workloads: noise

### Full 128-workload

| Group | Count | Mean (µs) | Max (µs) |
|---|---|---|---|
| Small (fast path) | 69 | 2.00 | 2 |
| Large (slow path) | 59 | 22.83 | 35 |
| Overall | 128 | **11.60** | 35 |

vs exp 28: 19.85 → 11.60 µs → **-42% full-run mean**.
vs exp 26 baseline: 0.0276 → 0.0116 ms → **-58% cumulative**.

## Learnings

1. **`num_warps=8` is the right knob for reduction-heavy kernels on large tiles.** The radix_topk_kernel does 32× `tl.sum` + 2× `tl.cumsum` over BLOCK_N=4096-8192. At default `num_warps=4` (128 threads), reductions are throttled by warp-level tree depth. Doubling to 8 warps (256 threads) halves per-warp work and keeps the reduction tree shallow. LESSONS.md had no prior evidence on this — exp 11, 24 tried num_warps on `score_kernel` (single MMA, no loops → no-op). This is the first kernel with real reduction work.
2. **Reduction-bound kernels scale like `tile_size / num_threads`, not `tile_size`.** At mp=91 (BLOCK_N=8192), num_warps=4 bottlenecked us. At num_warps=8, the kernel runs in ~35 µs total (across 29 concurrent programs on 132 SMs).
3. **Previous `num_warps` results (exp 5, 11, 24) were false-negatives in a specific sense.** Those kernels had a single `tl.dot` and no reductions — there was nothing to parallelize across warps. Don't generalize "num_warps doesn't help here" from MMA-only kernels to reduction-heavy ones.

## Next candidates

The radix kernel is now well-tuned for current work distribution. Slow-path mean is 22.8 µs. Further wins likely come from:
- **Fusing score_kernel + radix_topk_kernel** into one mega-kernel (save HBM round-trip on scores buffer + `torch.empty` overhead).
- **Per-batch scoreless inside radix**: already done — scoreless batches in the slow path emit natural-order and return early.
- **Early-termination of the bit loop** when count matches topk exactly (save ~5-15 bit iterations for many workloads).
- **Reducing `torch.empty` on scores**: use a module-level pool or skip if there's a way to compute in shared memory.
- **Tuning score_kernel `num_warps`**: its BLOCK_T=64 may benefit similarly, though exp 11 was negative (worth re-testing post-structural changes).
