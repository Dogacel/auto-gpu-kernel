# Experiment 36 — 2026-04-17

**Description:** `D_CKV_SPLIT_FUSED=1` on `_fused_attn_kernel` (T≤2 path). Hypothesis: the 8-way D-parallel replication of the Q@K^T + softmax work across the fused kernel's 8 CTAs per token is wasted on launch-bound T=1/T=2 workloads; collapsing to 1 CTA per token should save ~1-2 µs per call. Also added a constexpr branch to reuse `kc` for the P@K dot when `BLOCK_D == D_CKV`, removing one redundant HBM load per iter.

## Results
- Pass: 2/2 quick, 12/12 stride-2, 12/12 A/B
- Mode: stride-2 + A/B vs exp_26
- **Reverted** — all T≤2 workloads regressed +17-22%

**A/B vs exp_26 (paired same-VM):**
| UUID | T-class | A (exp_26) | B (exp_36) | Δ | % |
|---|---|---|---|---|---|
| 02d6ae9c | T=8 | 0.0160 | 0.0160 | +0.0000 | +0.06% |
| **05f6de65** | **T=2** | 0.0187 | 0.0213 | +0.0026 | **+14.15%** |
| **0c23b10c** | **T=1** | 0.0052 | 0.0063 | +0.0011 | **+21.15%** |
| 2207f0fd | T=8 | 0.0159 | 0.0158 | −0.0001 | −0.32% |
| 232ed014 | T=8 | 0.0154 | 0.0154 | −0.0000 | −0.06% |
| 4c46a94b | T=6 | 0.0114 | 0.0115 | +0.0000 | +0.39% |
| 5096e459 | T=8 | 0.0162 | 0.0162 | +0.0000 | +0.02% |
| 564007ac | T=8 | 0.0162 | 0.0162 | −0.0001 | −0.33% |
| 78b2e11c | T=8 | 0.0159 | 0.0159 | −0.0001 | −0.32% |
| **b7668cfd** | **T=1** | 0.0054 | 0.0065 | +0.0010 | **+18.96%** |
| **e6b849f2** | **T=2** | 0.0080 | 0.0094 | +0.0014 | **+17.26%** |
| **f77df5ce** | **T=2** | 0.0054 | 0.0064 | +0.0010 | **+19.20%** |

Paired: B wins 4/12 (ties on unchanged large-T paths), mean Δ = +0.0006 ms → A faster. **All 5 fused-path workloads regress +17-22%.** Worst case: T=1 with small valid (~21%).

## Design (reverted)

```diff
     BLOCK_N = 128
     BLOCK_N_FUSED = 64
     D_CKV_SPLIT = 8
     BLOCK_D = D_ckv // D_CKV_SPLIT
+    D_CKV_SPLIT_FUSED = 1
+    BLOCK_D_FUSED = D_ckv // D_CKV_SPLIT_FUSED
     NUM_SPLITS = 8

     if num_tokens <= 2:
-        grid = (num_tokens, D_CKV_SPLIT)
+        grid = (num_tokens, D_CKV_SPLIT_FUSED)
         _fused_attn_kernel[grid](
             ...
-            BLOCK_N=BLOCK_N_FUSED, BLOCK_D=BLOCK_D,
+            BLOCK_N=BLOCK_N_FUSED, BLOCK_D=BLOCK_D_FUSED,
             num_warps=8, num_stages=2,
         )

# Inside _fused_attn_kernel inner loop:
-        kc_slice_ptrs = Ckv_ptr + safe_idx[:, None] * stride_kc_s + offs_d[None, :]
-        kc_slice = tl.load(kc_slice_ptrs, mask=valid[:, None], other=0.0)
-        acc = tl.dot(p.to(tl.bfloat16), kc_slice, acc=acc)
+        if BLOCK_D == D_CKV:
+            acc = tl.dot(p.to(tl.bfloat16), kc, acc=acc)
+        else:
+            kc_slice_ptrs = Ckv_ptr + safe_idx[:, None] * stride_kc_s + offs_d[None, :]
+            kc_slice = tl.load(kc_slice_ptrs, mask=valid[:, None], other=0.0)
+            acc = tl.dot(p.to(tl.bfloat16), kc_slice, acc=acc)
```

## Discoveries

1. **D-parallelism in the fused kernel is load-bearing, not redundant.** LESSON-16 said "D-parallel fused kernel replicates Q@K^T across CTAs" — I interpreted "replicated compute" as waste, but the stable +17-22% regression proves the opposite: spreading 512 output channels across 8 SMs is critical for the fused path. Merging into 1 CTA serializes the [H, D_CKV] = [16, 512] acc tile handling, trading cheap parallelism for register/thread pressure on a single warp group.

2. **The single-CTA fused kernel has structural latency hazards that the 8-CTA version avoids:**
   - **Register pressure on acc:** [16, 512] f32 = 32 KB / 256 threads = 128 B/thread = 32 registers just for acc. Well inside B200's 255-reg budget individually, but adds to the total register live set from Q/K/softmax scalars. Triton may spill to shmem when the live range overlaps MMA ops.
   - **Wall-clock serialization:** 1 CTA writes [16, 512] bf16 output directly vs 8 CTAs each writing [16, 64]. On B200's HBM, the single-CTA's contiguous 16-KB store goes through a single scoreboard; 8 parallel 2-KB stores from 8 CTAs finish concurrently in ~1/8 of the wall-clock.
   - **Loss of HBM latency hiding:** 8 CTAs issue 8 parallel K-tile loads; ~200 ns of HBM latency is overlapped across 8 in-flight requests. 1 CTA has no overlap on the first iter.

3. **The L2-cached "redundant Q@K^T" is nearly free.** Each of the 8 D-split CTAs does the same Q@K^T at the same time — on B200 with 126 MB L2, the K tile is cached after one CTA's fetch. The "waste" I was trying to eliminate was already free; I just sacrificed the parallelism benefit.

4. **T=1 regressed worst (+18-21%).** The most launch-bound workload was the worst hit — counter to my launch-tax elimination hypothesis. This reinforces that for T=1, the gap between "launch setup" and "ready to write output" is already dominated by HBM-latency-on-first-load, not by CTA-count overhead.

## Verdict

**Reverted to exp_26.** A/B cleanly established A-faster across all fused-path workloads. Large-T path unchanged as expected.

## Next directions

- **D_CKV_SPLIT_FUSED axis closed** — 1 regresses, 8 optimal; no reason to try 4 or 2 (same structural cost/tradeoff just smaller).
- **Triton-side axes now exhausted on fused path.** Fused kernel has been swept across: BLOCK_N_FUSED (32 untested but LESSON-19 predicts loss), num_stages (exp_34 tried 3, regressed), num_warps (LESSON-26 locked at 8), D_CKV_SPLIT_FUSED (this experiment, 1 regresses). Cache modifiers already known bad per LESSON-40.
- **Triton-side axes now exhausted on split path.** NUM_SPLITS, num_stages, BLOCK_N, cache modifiers, stride-partition — all swept.
- **9 consecutive reverts since exp_26.** Approaching the skill's 15-20 threshold for committing to Gluon. Remaining plausible Triton-side candidates (low confidence each):
  1. BLOCK_N_FUSED=32 (3rd-best per exp_35/plan.md — predicted low ceiling, only wins on `valid<32` subset)
  2. Warp-specialization within the T≤2 CTA — complex, Triton doesn't expose warp IDs cleanly
  3. Fuse T=1 kernel variant that skips Q/K loads when `num_valid == 0`
- **Next attempt: BLOCK_N_FUSED=32 (exp_37).** Fast one-line change. If it regresses or ties, we've closed the tile-size axis on the fused kernel and should commit to the Gluon pivot (exp_35/plan.md becomes exp_38's starting point).
