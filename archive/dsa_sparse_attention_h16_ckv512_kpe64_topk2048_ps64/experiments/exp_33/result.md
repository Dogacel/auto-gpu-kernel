# Experiment 33 — 2026-04-17

**Description:** `BLOCK_N = 256` on split kernel (double from 128). Hypothesis: under stride-partition SPLIT_SIZE=256, max_bn ≤ 256 so loop runs ≤1 iter always; larger tile removes loop overhead.

## Results
- **Compile failure: shmem OOM.** Required 321536 B, hardware limit 232448 B.
- Quick: T=1 passed (unaffected, different kernel); T=8 RUNTIME_ERROR
- Mode: compile-fail; no measurement
- **Reverted**

## Design (reverted)

```diff
-    BLOCK_N = 128
+    BLOCK_N = 256
```

## Discoveries

1. **BLOCK_N=256 exceeds B200's shmem budget under num_stages=2.** K tile is `BLOCK_N × (D_ckv + D_kpe) × 2 = 256 × 576 × 2 = 288 KB` per stage. Two stages = 576 KB > 232 KB shmem/SM. Triton's shmem planner requires 321 KB (closer to 1.4× of one tile — some overhead + partial allocation).

2. **BLOCK_N=256 is infeasible without dropping num_stages.** num_stages=1 halves shmem need to ~160 KB (feasible) but exp_32 showed num_stages=1 regresses T=8 by +62% on the existing BLOCK_N=128 config. Combining both changes would confound the measurement and likely regress.

3. **The BLOCK_N ↔ num_stages coupling is a hard wall.** To get BLOCK_N=256 working, need either:
   - num_stages=1 (regressed in exp_32, so no)
   - Smaller D (would change kernel semantics)
   - Gluon with explicit shmem control (can hand-pack tiles tighter)
   - Persistent kernel that reuses shmem across iters differently
   None of these are one-line changes.

4. **B200 shmem size = 232 KB/SM** confirmed from the Triton error. Matches public specs.

## Verdict

**Reverted.** Compile failure, no measurement.

## Next directions

- BLOCK_N axis on split kernel now bounded at 128 by shmem constraint with num_stages=2.
- **Pivot options:**
  1. **Gluon rewrite of the MMA path** with `bw.tcgen05_mma` (tensor-core MMA, not software FMA like exp_22). Explicit shmem control would let BLOCK_N=256 fit. High-risk, 10+ iter runway, but unlocks the shmem wall.
  2. **Fused T≤2 kernel tuning** — `num_stages` and `BLOCK_N_FUSED` not fully swept on that kernel. exp_13 was BLOCK_N_FUSED=64; exp_32 tested num_stages=1 on split only. Try num_stages=3 on fused or BLOCK_N_FUSED=32 on fused.
  3. **Launch-overhead attack**: grid reduction for T=1 case. Currently (1, 8) = 8 CTAs. Can we reduce to 4 or 2 by increasing BLOCK_D on fused?
- Best next small experiment: **exp_34 test `num_stages=3` on fused T≤2 kernel** — distinct from split kernel, iterates up to 32 times, may benefit from deeper pipelining.
