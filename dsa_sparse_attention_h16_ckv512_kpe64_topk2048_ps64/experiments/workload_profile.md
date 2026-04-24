# Workload Profile
_Generated 2026-04-16 from 23 workloads of `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`._

## Summary (top 5 actionable facts)

1. **T is always ≤ 8** — distribution is `{1: 1, 2: 8, 6: 3, 7: 3, 8: 8}`; max T = 8, median T = 6. The grid is always starved of programs: even at T=8, we launch only 8 CTAs on a 148-SM B200 → **split the TopK reduction across many CUDA programs** (flash-decoding). `NUM_SPLITS=16` gives `8×16 = 128` CTAs at T=8 and `1×16 = 16` at T=1.
2. **Most tokens use very few KV entries.** Padding distribution over 120 tokens: `p10=959, p50=2015, p90=2040, max=2047` out of 2048. Median token has only **33 valid entries**; 88.3% of tokens have `pad > 1024`. A single BLOCK_N=64 would already cover the median; processing all 2048 topk slots is **~60× wasted work** on the typical token.
3. **Valid entries are contiguous, not scattered.** Per-token contiguity (fraction of consecutive-valid pairs with `idx[i+1]==idx[i]+1`) has `p50=1.0, mean=0.96`. A typical token's 33 valid entries cluster into **1 page** of 64 (`unique_pages p50 = 1, p90 = 18, max = 43`). The indices look like a prefix of consecutive KV positions followed by `-1` padding → **an early-exit when a whole BLOCK_N block is all `-1`** skips nearly all the work on small tokens.
4. **P is fixed at 8462 across all 23 workloads** — constant KV-cache size. Don't parameterize grid/tile sizing on P; no need for dynamic strategy by cache size.
5. **Cross-token page reuse ≈ 1.0.** Each referenced page is read by ~1 token (mean 1.005, p90 1.01). So a shared-SMEM KV tile across query tokens buys almost nothing; split-K on the **TopK axis** (per-token) is the right axis, not merging work across tokens.

## Distributions

### num_tokens (T)
- `min=1, p10=2, p50=6, p90=8, max=8, mean=5.22, n=23`
- Histogram: `[1,2): 1 | [2,4): 8 | [4,8): 6 | [8,16): 8`
- Exact counts: `T=2 (8 workloads), T=8 (8 workloads), T=6 (3), T=7 (3), T=1 (1)`
- **All workloads fit in one of two "modes": low (T ≤ 2, 9 workloads) or batched (T ∈ {6,7,8}, 14 workloads).**

### num_pages (P)
- Every workload: `P = 8462`. No variation.
- `ckv_cache` is `[8462, 64, 512] bf16` = 275 MiB; `kpe_cache` is 35 MiB → always HBM-resident.

### sparse_indices padding (-1)
- Per-token padding count across 120 tokens: `min=0, p10=959, p50=2015, p90=2040, max=2047`.
- Valid count: `min=1, p10=8, p50=33, p90=1089, max=2048`.
- Fraction of tokens with any padding: **96.7%**. Fully-valid (pad==0): 3.3%. Mostly-invalid (pad>1024): **88.3%**.
- The shape is bimodal: most tokens have O(10–100) valid entries, a minority have O(1000–2048).

### sparse_indices structure
- Contiguity: fraction of `idx[i+1]==idx[i]+1` among valid pairs per token: `p10=0.99, p50=1.00, p90=1.00, mean=0.96`.
- Unique pages per token (`len(unique(valid // 64))`): `min=1, p10=1, p50=1, p90=18, max=43`.
- Interpretation: the valid portion is usually a **contiguous run of KV positions**, padded with `-1` to 2048. Small tokens fit in a single 64-wide page; only a few (workloads 9–22, the batched ones) touch >10 pages for some of their tokens.
- Cross-token page reuse (mean count of tokens referencing the same page, per workload): `p50=1.00, p90=1.01, max=1.06`. Tokens in a batch rarely share KV pages.

### Total workloads
- **23 workloads** (not 24). `stride=2 → 12 workloads` is consistent (ceil(23/2)=12). The `quick` mode picks `workloads[0]` and `workloads[-1]` (T=1 and T=7) — intentionally spans the range.

## Recommended optimizations (priority ordered)

1. **Split-K on the TopK axis (flash-decoding style).**
   **What:** launch grid `(T, NUM_SPLITS)`, each program computes partial `m_i, l_i, acc` over `TOPK // NUM_SPLITS` indices; second reduction kernel merges splits.
   **Why:** T ≤ 8 in every workload; B200 has 148 SMs. With `NUM_SPLITS=16`, `T=1 → 16 CTAs`, `T=8 → 128 CTAs` (≈ full occupancy). Without split, `T=1` uses 1 SM.
   **Impact:** all 23 workloads (100%). Speedup dominated by T=1,2 (9 workloads, ~39%) where current scheme is 1–2 CTAs; expect 4–8× on those, 1.5–2× on T=6..8.

2. **Early-exit on all-`-1` TopK blocks.**
   **What:** inside the main loop `for bn in range(0, TOPK, BLOCK_N)`, load indices first, `if tl.max(idx) < 0: continue` (skip the two dots and the softmax update).
   **Why:** p50 valid count is 33 → after the first ~1 BLOCK_N (64 wide), **31 of 32 blocks are all-padding**. Each skipped block avoids `(16×64×512) + (64×16×512)` FMAs.
   **Impact:** ~88% of tokens (pad>1024) see roughly `valid_blocks / 32 ≈ 5-10%` of current work; **10–20× on those tokens' inner loop**. Combined with split-K, dominates latency on small-valid-count workloads (workloads 0–8, T∈{1,2}, valid p50 ≈ 2–172). Note: may reduce benefit of split-K on T=1 workloads.

3. **Tile-aligned page load (contiguous prefetch).**
   **What:** when `tl.max(idx) - tl.min(idx) < BLOCK_N` and all valid, load the KV slice as a single contiguous block starting at `tl.min(idx)` instead of 64 individual gathers.
   **Why:** contiguity p50=1.0, and typical tokens fit in 1 page of 64. The gather `Ckv_ptr + idx[:, None]*stride + offs` forces 64 independent address computations per block; a strided load from a known base gets coalesced better.
   **Impact:** p90=18 pages → up to 20% of tokens (the batched workloads 18–22) spread over multiple pages; others are near-single-page. Expect 10–20% on the inner loop where applicable. Lower priority than 1 & 2 because Triton's gather often already coalesces consecutive indices on B200.

4. **Specialize T=1 path (single-token kernel).**
   **What:** compile-time branch on `T==1` that drops outer loop, maximizes split count (`NUM_SPLITS=32`).
   **Why:** 1 workload (~4.3%) but it is the smallest and quickest to regress. After split-K is in place, the T=1 latency is determined by reduction overhead; keeping the outer loop complexity hurts it disproportionately.
   **Impact:** 1/23 workloads; nice-to-have once split-K exists.

## Notes
- Q/KV tensors (`q_nope`, `q_pe`, `ckv_cache`, `kpe_cache`) are declared as `RandomInput` in the workload spec — **real numerical distributions are not available** from the trace set. Value-sparsity and scalar-magnitude-based optimizations (skip-zero weights, etc.) cannot be evaluated offline.
- `sm_scale` varies per workload (sample value seen: `0.1352337788608801`), as expected for MLA's `1/sqrt(192)` scaling. No specialization possible.
- Inspection script: `scripts/inspect_workloads.py` (`modal run scripts/inspect_workloads.py`), full JSON at `/tmp/claude-1000/modal_logs/full_result.json` (ephemeral).

---

## Stride-2 NUM_SPLITS iter-count analysis (added 2026-04-17)

**Goal:** understand why exp_51 (`NUM_SPLITS=8→16`) regressed `4c46a94b` (+14%) while winning on 6 other T=7/8 workloads. Informs adaptive NUM_SPLITS dispatch (exp_52 candidate).

**Derivation.** Kernel uses stride-partition `offs = s + arange(SPLIT_SIZE)*NUM_SPLITS`. Indices are (per profile) contiguous-valid-prefix then `-1` padding (`contig_frac_mean ≈ 1.0`). For per-token valid count `N` with `NUM_SPLITS=S`:
- Per-split-valid for split 0 (worst case) = `ceil(N / S)`; for split s < S: `ceil((N-s)/S)`
- `max_bn = ceil(per_split_valid / BLOCK_N) * BLOCK_N`, BLOCK_N=128
- Split-phase iters = `max_bn / BLOCK_N`
- Grid `(T, NUM_SPLITS)` latency ≈ slowest CTA = worst-case token × worst-case split

At NUM_SPLITS=8, iters jumps 1→2 when per-token valid crosses **1024** (ceil(1024/8)=128, 1 iter; ceil(1025/8)=129, 2 iters). At NUM_SPLITS=16, boundary is **2048** (ceil(2048/16)=128, still 1 iter; only ceil(2049/16)=129 would be 2 iters, but TOPK=2048 caps it). So at NUM_SPLITS=16, **every** split fits in 1 iter regardless of valid count.

### Table: stride-2 workloads, valid-count distribution, and split-iter impact

| idx | uuid | T | valid min/med/max | per-split max @8 | per-split max @16 | iters @8 | iters @16 | path | exp_51 Δ |
|---:|:---|---:|:---|---:|---:|---:|---:|:---|:---|
| 0 | `0c23b10c` | 1 | 2 / 2 / 2 | 1 | 1 | 1 | 1 | FUSED | ≈0 |
| 2 | `b7668cfd` | 2 | 33 / 42 / 52 | 7 | 4 | 1 | 1 | FUSED | ≈0 |
| 4 | `05f6de65` | 2 | 6 / 171 / 337 | 43 | 22 | 1 | 1 | FUSED | +0.3-0.8% |
| 6 | `e6b849f2` | 2 | 48 / 70 / 92 | 12 | 6 | 1 | 1 | FUSED | ≈0 |
| 8 | `f77df5ce` | 2 | 18 / 18 / 19 | 3 | 2 | 1 | 1 | FUSED | ≈0 |
| 10 | **`4c46a94b`** | **8** | **2 / 21 / 1002** | **126** | **63** | **1** | **1** | **split** | **+14.4%** |
| 12 | `02d6ae9c` | 8 | 6 / 37 / 2048 | 256 | 128 | **2** | **1** | split | -15.8 to -19.5% |
| 14 | `78b2e11c` | 8 | 11 / 35 / 2048 | 256 | 128 | **2** | **1** | split | -15.8 to -19.3% |
| 16 | `564007ac` | 8 | 4 / 212 / 2048 | 256 | 128 | **2** | **1** | split | -15.8 to -19.3% |
| 18 | `232ed014` | 8 | 1 / 44 / 1091 | 137 | 69 | **2** | **1** | split | -15.8 to -19.3% |
| 20 | `5096e459` | 8 | 1 / 98 / 1986 | 249 | 125 | **2** | **1** | split | -15.8 to -19.3% |
| 22 | `2207f0fd` | 7 | 131 / 263 / 2011 | 252 | 126 | **2** | **1** | split | -15.8 to -19.3% |

valid = `2048 - pad_{min,p50,max}`. per-split max = `ceil(valid_max / NUM_SPLITS)`. iters = `ceil(per_split_max / 128)`.

### `4c46a94b` — the regressing workload

- **T=8, valid_max=1002** (pad_min=1046). The token with the most work has 1002 valid entries.
- At NUM_SPLITS=8: worst-case per-split = ceil(1002/8) = **126** → `max_bn=128` → **1 iter**.
- At NUM_SPLITS=16: worst-case per-split = ceil(1002/16) = **63** → `max_bn=128` → **1 iter**.
- **Iter count unchanged (1→1).** But combine phase does 16-way reduction instead of 8-way, doubling its cost, and CTA count doubles (8×8=64 → 8×16=128), doubling launch overhead on an already-1-iter split phase.
- This is a true loss case for NUM_SPLITS=16: `valid_max ∈ (1024/??, 1024]` range where doubling splits only doubles combine.

### Which workloads benefit from NUM_SPLITS=16 vs NUM_SPLITS=8

**Iter-reduction benefit (8→16 is a win):** `valid_max > 1024` on the split path. In stride-2: `02d6ae9c, 78b2e11c, 564007ac, 232ed014, 5096e459, 2207f0fd` (6/12). All have valid_max ∈ {1091, 1986, 2011, 2048, 2048, 2048}.

**No benefit (flat or regression, 8→16 adds combine overhead):** `valid_max ≤ 1024` on the split path. In stride-2: `4c46a94b` (valid_max=1002). Mechanism: both configs already do 1 iter; extra splits just split the combine.

**Untouched (fused path T≤2):** `0c23b10c, b7668cfd, 05f6de65, e6b849f2, f77df5ce` (5/12). Fused kernel doesn't take NUM_SPLITS.

### Adaptive dispatch rule (for exp_52)

Compute `valid_max = (indices >= 0).sum(axis=-1).max()` once on host (cheap — int32 scan over T×2048) and pick:
- `T <= 2` → fused path (no change).
- `T >= 3, valid_max > 1024` → NUM_SPLITS=16.
- `T >= 3, valid_max ≤ 1024` → NUM_SPLITS=8.

Tighter boundary: the exact crossover is `valid_max > 8 * BLOCK_N = 1024` for the 1↔2-iter jump at S=8. Anywhere at-or-below 1024 gets no iter-reduction benefit from doubling splits.

**Projected impact:** on stride-2, all 6 "iter-reduction" wins retained (-15-19% each) and the `4c46a94b` regression (+14%) is eliminated. Net: ~17 µs saved across the 12 workloads vs exp_51's ~15 µs.


