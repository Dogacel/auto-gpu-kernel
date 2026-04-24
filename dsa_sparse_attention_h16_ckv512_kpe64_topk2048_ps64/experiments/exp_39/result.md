# Experiment 39 — 2026-04-17

**Description:** Merged `partial_m` + `partial_l` → `partial_ml [num_tokens, NUM_SPLITS, H, 2]` (interleaved last axis). One `tl.store(pml_ptrs, tl.join(m_i, l_i))` in split phase; one `tl.load(pml_ptr_s)` + `tl.split(ml_si)` in combine phase. Plan: halve the 16 combine-phase scalar loads to 8 by cache-line-colocating m and l. Projected −0.2 to −1.0% on T≥3.

## Results
- Pass: 2/2 quick, 12/12 A/B ×2
- Mode: quick + A/B vs exp_37 (two runs for noise check per plan)
- **Reverted** — consistent +0.5-1.5% regression across T≥3 workloads

**A/B vs exp_37 (two paired runs):**

Run 1:
| UUID | T-class | A (exp_37) | B (exp_39) | Δ | % |
|---|---|---|---|---|---|
| 02d6ae9c | T=8 | 0.0155 | 0.0155 | +0.0000 | +0.02% |
| 05f6de65 | T=2 | 0.0186 | 0.0185 | −0.0001 | −0.48% |
| 0c23b10c | T=1 | 0.0052 | 0.0052 | −0.0000 | −0.06% |
| **2207f0fd** | **T=8** | 0.0154 | 0.0155 | +0.0001 | **+0.69%** |
| **232ed014** | **T=8** | 0.0151 | 0.0153 | +0.0002 | **+1.53%** |
| **4c46a94b** | **T=6** | 0.0110 | 0.0111 | +0.0001 | **+1.02%** |
| **5096e459** | **T=8** | 0.0156 | 0.0158 | +0.0002 | **+0.98%** |
| **564007ac** | **T=8** | 0.0157 | 0.0158 | +0.0002 | **+1.02%** |
| **78b2e11c** | **T=8** | 0.0153 | 0.0154 | +0.0002 | **+0.98%** |
| b7668cfd | T=1 | 0.0054 | 0.0054 | +0.0000 | +0.35% |
| e6b849f2 | T=2 | 0.0079 | 0.0079 | −0.0000 | −0.44% |
| f77df5ce | T=2 | 0.0053 | 0.0053 | +0.0000 | +0.24% |

B wins 3/12, mean Δ = +0.0001 ms. 232ed014 breaches +1% hard-revert threshold.

Run 2 (confirmation):
| UUID | T-class | A (exp_37) | B (exp_39) | Δ | % |
|---|---|---|---|---|---|
| 02d6ae9c | T=8 | 0.0154 | 0.0155 | +0.0001 | +0.56% |
| 05f6de65 | T=2 | 0.0187 | 0.0188 | +0.0001 | +0.50% |
| 0c23b10c | T=1 | 0.0055 | 0.0056 | +0.0001 | +0.99% |
| **2207f0fd** | **T=8** | 0.0156 | 0.0158 | +0.0002 | **+1.32%** |
| **232ed014** | **T=8** | 0.0151 | 0.0152 | +0.0002 | **+1.08%** |
| **4c46a94b** | **T=6** | 0.0109 | 0.0111 | +0.0001 | **+1.26%** |
| 5096e459 | T=8 | 0.0157 | 0.0158 | +0.0001 | +0.55% |
| 564007ac | T=8 | 0.0158 | 0.0159 | +0.0001 | +0.73% |
| 78b2e11c | T=8 | 0.0154 | 0.0155 | +0.0002 | +0.98% |
| b7668cfd | T=1 | 0.0055 | 0.0055 | +0.0000 | +0.12% |
| e6b849f2 | T=2 | 0.0080 | 0.0080 | +0.0000 | +0.08% |
| f77df5ce | T=2 | 0.0054 | 0.0055 | +0.0000 | +0.77% |

B wins 0/12, mean Δ = +0.0001 ms. Regression confirmed.

## Design (reverted)

```diff
-    Partial_m_ptr, Partial_l_ptr, Partial_acc_ptr,
+    Partial_ml_ptr, Partial_acc_ptr,
     ...
-    stride_pm_t, stride_pm_s, stride_pm_h,
-    stride_pl_t, stride_pl_s, stride_pl_h,
+    stride_pml_t, stride_pml_s, stride_pml_h, stride_pml_c,

# Split-phase stores:
-    pm_ptrs = ... Partial_m_ptr + ...
-    pl_ptrs = ... Partial_l_ptr + ...
-    tl.store(pm_ptrs, m_i, cache_modifier=".cg")
-    tl.store(pl_ptrs, l_i, cache_modifier=".cg")
+    offs_c = tl.arange(0, 2)
+    pml_ptrs = ... (last dim = offs_c[None, :] * stride_pml_c)
+    ml_i = tl.join(m_i, l_i)
+    tl.store(pml_ptrs, ml_i, cache_modifier=".cg")

# Combine-phase loads:
-    m_si = tl.load(pm_ptr_s)
-    l_si = tl.load(pl_ptr_s)
+    ml_si = tl.load(pml_ptr_s)
+    m_si, l_si = tl.split(ml_si)

# Host allocation:
-    partial_m = torch.empty((num_tokens, NUM_SPLITS, H), ...)
-    partial_l = torch.empty((num_tokens, NUM_SPLITS, H), ...)
+    partial_ml = torch.empty((num_tokens, NUM_SPLITS, H, 2), ...)
```

## Discoveries

1. **`tl.join`/`tl.split` in the hot path add Triton-codegen overhead that exceeds cache-locality gains.** The interleaved `[H, 2]` layout places m[h] and l[h] as adjacent scalars — one cache line fetch retrieves both. But `tl.join(m_i, l_i)` likely adds an additional register shuffle in the split-phase write, and `tl.split(ml_si)` adds a similar de-interleave on the combine side. At H=16 (small) the shuffle overhead dominates the saved cache-line fetch.

2. **Contrast with the monotonic-counter win:** exp_37 removed actual work (one atomic RMW per call). This change merely reshuffled the same data. Removing work always wins when the removal is free; reshuffling to be "more efficient" only wins when the compiler can't already schedule the original layout well — and here Triton was already scheduling the two adjacent (pm, pl) loads without measurable overhead.

3. **Original `Partial_m` and `Partial_l` were both L2-resident already.** At NUM_SPLITS=8 × num_tokens=8 × H=16 × 4B = 4 KB each tensor total. They easily fit in L2. Whether adjacent in memory or not, the cache miss rate on the second load is near zero. The "cache line colocation" benefit was over-theorized.

4. **Plan's +0.2 to −1.0% projection was optimistic.** Plan assumed compiler would fuse two scalar loads into one vectorized load. In practice Triton's cross-tensor fusion is already aggressive — the explicit join/split reorg doesn't unlock additional fusion.

5. **Iterating on the combine-IO axis is now closed.** Exp_28 (`.cg` on combine loads: regression), exp_30 (buffer persistence: neutral), exp_39 (interleave: regression). Three independent attempts all ≤ ties. Combine phase is at its Triton optimum.

## Verdict

**Reverted to exp_37.** Kernel is exp_37-baseline: monotonic counter, stride-partition, `.cg` modifiers as established.

## Next directions

- **Combine-IO axis closed.** exp_28, exp_30, exp_39 all neutral/regression. Moving on.
- **Per plan's "exp_40 = Gluon go/no-go" coordination note:** Exp_40 should restart from `experiments/exp_35/plan.md` (preserved Gluon pivot blueprint, stage 1: `bw.tcgen05_mma` bring-up on split path). Justifications:
  1. All cheap Triton scalar axes closed (LESSONS 12, 19, 26, 27, 29, 40, 41, 43, 45).
  2. All cheap Triton combine-IO axes closed (exp_28, exp_30, exp_39).
  3. The 1.8 µs remaining barrier-spin requires `bw.mbarrier` or cluster-sync — **Gluon-only primitives**.
  4. exp_35 plan's self-projected +5-20% regression on stage 1 is acceptable per /optimize skill's "15-20 plateau" threshold; we're at 13 iterations since exp_26, within range.
- **Alternative axis to consider in parallel:** warp-specialization inside `_fused_attn_kernel` (T≤2 path). Never tried. Plan's "fallback-to-fallback." Requires `tl.inline_asm` — brittle. Likely parked unless Gluon path stalls.
- **Next attempt: exp_40 — Gluon stage 1 per exp_35/plan.md.** Delegate to fresh sub-agent with clean context. Accept +5-20% regression as investment. If exp_40 passes correctness + stays under 0.025 ms, continue to exp_41 (`bw.mbarrier`). If exp_40 OOM or can't compile, fall back to exp_41-as-T=1-warp-split.
