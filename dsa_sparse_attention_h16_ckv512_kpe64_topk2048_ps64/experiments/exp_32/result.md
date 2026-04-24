# Experiment 32 — 2026-04-17

**Description:** `num_stages=1` on `_fused_split_combine_kernel` (down from 2). Hypothesis: under stride-partition, most workloads run only 1 iter of the split loop (num_valid per split ≈ total/NUM_SPLITS), so num_stages=2 pipelining overhead may be pure cost.

## Results
- Pass: 2/2 quick
- Mode: quick only — did not proceed to A/B
- **Reverted** — T=8 regressed +62% (same magnitude as exp_31)

**Quick data:**
| UUID | exp_26 (num_stages=2) | exp_32 (num_stages=1) | Δ |
|---|---|---|---|
| 0c23b10c (T=1) | 0.005 ms | 0.005 ms | — (fused path unaffected) |
| 2207f0fd (T=8) | 0.016 ms | 0.026 ms | +62% |

## Design (reverted)

```diff
-        num_warps=8, num_stages=2,
+        num_warps=8, num_stages=1,
```

## Discoveries

1. **T=8 is multi-iter per split**, not 1-iter as the hypothesis assumed. For the T=8 workload 2207f0fd, `num_valid_total` per token is likely in the 500-1500 range (typical DSA prefix-valid distribution). Under stride-partition with NUM_SPLITS=8, per-split valid is 60-190 → ceil(valid/128) = 1-2 iters; on high-end, 2 iters.

2. **num_stages=2 double-buffers K tiles across iter boundary.** When iter-count ≥ 2, the pre-fetch of iter 1's K tile overlaps with iter 0's MMA. Killing that overlap (num_stages=1) serializes K-load with compute, hitting HBM latency twice instead of hiding it.

3. **Sweet spot at num_stages=2 confirmed across the full sweep**: exp_19 tested =3 (regress), exp_32 tests =1 (regress). Only =2 wins. Closed axis.

4. **num_stages is invisible to the single-iter case.** T=1/T=2 workloads (fused path) were unaffected because they don't use `_fused_split_combine_kernel`. Small-T path still has `num_stages=2` on `_fused_attn_kernel` — untested.

## Verdict

**Reverted to exp_26.** Fast quick result; no A/B needed.

## Next directions

- **num_stages axis on split kernel closed.**
- Remaining untouched knobs on split kernel:
  - `num_warps=8` — LESSON-26 says strict optimum at H=16; don't re-try.
  - `BLOCK_N=128` — tuned in exp_9 (128 > 64). Untested up-direction (256). Risk: larger K tile → shmem pressure → potential spill.
  - `cache_modifier` — LESSON-40 says axis saturated.
- Remaining untouched knobs on fused (T≤2) kernel:
  - `num_stages=2` — untested to sweep down to 1 or up to 3. The fused path iterates over full TopK in BLOCK_N_FUSED=64 chunks (up to 32 iters for fully-valid input) — should benefit from pipelining.
  - `BLOCK_N_FUSED=64` — tuned in exp_13 (64 > 128). Not swept to 32.
- **Next: exp_33 test num_stages=3 on fused T≤2 kernel.** The fused kernel iterates many more times than split under stride, so deeper pipelining might help; shmem budget is smaller in fused (smaller K tile per CTA) so num_stages=3 shouldn't blow shmem like exp_19 did on split.
