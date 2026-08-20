# Experiment 5 — num_warps=8 for score kernel

## Goal

Current kernel uses Triton's default num_warps (likely 4). Try
num_warps=8 to overlap more memory loads with the FP8 MMA.

## Rationale

- Score kernel does a single `tl.dot((64,128) x (128,64)) → (64,64)`
  per program, plus loads of Q (8 KB), K (8 KB), scales (256 B),
  weights (256 B), and a sum-reduction.
- With num_warps=4 (one warp group = 128 threads), all MMA-unit
  throughput comes from that warp group. A single (64,64) fp8 MMA
  finishes in ~100 cycles — small. Latency of K/scale loads may
  not be hiding well.
- num_warps=8 provides two warp groups; on Blackwell, this allows
  concurrent async memory operations via wgmma, potentially hiding
  K-load latency behind MMA.
- No loop in kernel, so num_stages doesn't apply; only num_warps.

## Change

```diff
     score_kernel[grid](
         ...,
         BLOCK_H=H, BLOCK_D=D, BLOCK_T=page_size,
+        num_warps=8,
     )
```

## Risk

- For small per-program work, more warps = more context switches per
  SM scheduler = potentially more scheduling overhead.
- Register pressure could increase; if spilling happens, performance
  tanks. Unlikely given compute is small and tile is square.

## Success criterion

- A/B benchmark paired same-VM: B wins at least 10/16 workloads, mean
  Δ ≤ −2% on stride 8. If less, try num_warps=2 instead (smaller tile
  per warp could fit better).
- Correctness still exact.
