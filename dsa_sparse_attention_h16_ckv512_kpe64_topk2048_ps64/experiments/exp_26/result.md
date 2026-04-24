# Experiment 26 — 2026-04-17

**Description:** Stride-partition split-phase TopK index distribution. Split `s` owns TopK positions `{s, s+NUM_SPLITS, s+2*NUM_SPLITS, ...}` instead of the contiguous block `[s*SPLIT_SIZE : (s+1)*SPLIT_SIZE)`. Distributes prefix-valid runs evenly across all 8 split CTAs so max-CTA wall time drops on small/medium-valid workloads.

## Results
- Pass: 23/23 full + 2/2 quick
- Max abs err: 1.56e-02 (unchanged)
- Mode: quick + full + 2 × stride-2 A/B paired

**A/B run 1 vs exp_24** (stride-2):
| UUID | T-class | A (exp_24) | B (exp_26) | Δ |
|---|---|---|---|---|
| 02d6ae9c | T=8 | 0.0155 | 0.0157 | +1.38% |
| 05f6de65 | T=2 | 0.0186 | 0.0186 | -0.19% |
| 0c23b10c | T=1 | 0.0055 | 0.0055 | +0.82% |
| 2207f0fd | T=8 | 0.0158 | 0.0159 | +0.08% |
| 232ed014 | T=8 | 0.0154 | 0.0154 | +0.25% |
| **4c46a94b** | **T=6** | **0.0154** | **0.0113** | **−26.66%** |
| 5096e459 | T=8 | 0.0158 | 0.0160 | +1.17% |
| 564007ac | T=8 | 0.0159 | 0.0160 | +0.76% |
| 78b2e11c | T=8 | 0.0155 | 0.0156 | +1.08% |
| b7668cfd | T=1 | 0.0054 | 0.0054 | +0.53% |
| e6b849f2 | T=2 | 0.0080 | 0.0079 | −0.84% |
| f77df5ce | T=2 | 0.0054 | 0.0054 | +0.72% |

Paired: 3/12 B wins, mean Δ = -0.0003 ms → B faster (driven entirely by the 4c46a94b outlier).

**A/B run 2 vs exp_24** (second VM): 7/12 B wins, mean Δ = -0.0003 ms (same direction). 4c46a94b again -25.27%. Reproduces.

**Full benchmark** (23 workloads): two structural wins confirm — `4c46a94b` T=6 = 0.012 ms and `68d6817d` T-class = 0.011 ms (the latter not in stride-2 subset). Other large-T workloads within ±1.5% of exp_24.

## Design

One 5-line edit inside `_fused_split_combine_kernel`:

```diff
-    start = s * SPLIT_SIZE
-    offs_split = tl.arange(0, SPLIT_SIZE)
-    idx_scan = tl.load(Indices_ptr + t * stride_idx_t + start + offs_split)
+    offs_split = s + tl.arange(0, SPLIT_SIZE) * NUM_SPLITS
+    idx_scan = tl.load(Indices_ptr + t * stride_idx_t + offs_split)
     num_valid = tl.sum((idx_scan >= 0).to(tl.int32), axis=0)
     max_bn = ((num_valid + BLOCK_N - 1) // BLOCK_N) * BLOCK_N

     for bn in range(0, max_bn, BLOCK_N):
-        idx_ptrs = Indices_ptr + t * stride_idx_t + (start + bn + offs_n)
+        idx_ptrs = Indices_ptr + t * stride_idx_t + (s + (bn + offs_n) * NUM_SPLITS)
```

Combine phase and the atomic barrier are untouched — `partial_*[t, s, ...]` layout stays the same. Only the mapping from split-index `s` → TopK positions changes.

## Mechanism

Under block-partition, a prefix-valid run of length `N` concentrates work on `ceil(N/SPLIT_SIZE)` splits (the rest exit via `max_bn=0`). For `N=33` (workload median), split 0 does **all** the work; splits 1–7 idle. Max-CTA wall time = split 0's compute.

Under stride-partition, split `s` sees ~`N/NUM_SPLITS` valid entries drawn from the prefix. For `N=33`, each split has 4–5 valid → each runs exactly 1 BLOCK_N=128 iter in parallel. For `N<BLOCK_N` the wall time is identical (both 1 iter). The win appears when **max-per-token valid** falls in `[~200, 1024]`: block-partition needs 2 iters, stride-partition still fits in 1. Above ~1024 both need 2 iters (saturated). Above ~2000 strided HBM coalescing loses vs block (tiny regression).

## Discoveries

1. **Stride-partition is a real structural win on a specific workload regime** (max-per-token valid in mid-hundreds). On `4c46a94b` (T=6) this is a reproducible -26% win across 3 measurements (A/B ×2 + full). Biggest single-workload win since exp_11's hybrid-dispatch.

2. **Large-valid T=8 workloads regress slightly** (~1% on 4/7 large-T workloads). Attributed to HBM coalescing loss: block-partition fetches 128 contiguous KV rows per iter, strided fetches 128 rows at stride 8. Under HBM burst semantics, strided is marginally less efficient. L2 caching hides most of it (rows land in L2 across splits, subsequent CTAs hit L2) but not all.

3. **The num_valid-based `max_bn` bound still holds under stride-partition** because valid entries remain contiguous in `idx_scan` — for prefix-valid input of length N, strided split `s` sees valid at `idx_scan[0..ceil((N-s)/NUM_SPLITS))`. Invariant preserved.

4. **Correctness is bit-robust**: strided reorders the `partial_acc`'s contribution per-split but combine still receives all NUM_SPLITS partials and merges via online softmax (order-independent up to fp32 associativity). Max abs_err unchanged (1.56e-02).

## Verdict

**Kept, new best.** Mean Δ favors B in both A/B runs. Single reproducible -26% win. Rest within ±1.5%. Net: cleanest structural win since exp_15.

Trade-off acknowledged: slight regression on ~4 large-T workloads. Acceptable given the outlier magnitude.

## Next directions

- **Hybrid partition**: only stride when `max_per_token_num_valid < threshold`. Need in-kernel check; adds a branch. Worth pursuing if the large-T regressions become material.
- **Investigate HBM coalescing regression**: try `.cg` vs default on the strided K/P loads to see if L1 caching changes the picture.
- **Apply strided partition to the small-T `_fused_attn_kernel`**: the D-split grid `(T, D_CKV_SPLIT=8)` has D-programs that don't split TopK — but each program still loops the full TopK. Not applicable; the strided pattern is a split-K win, not a D-split win.
