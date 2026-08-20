# Workload Profile
_Generated 2026-04-17 from all 128 workloads of `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`._
_Source: `scripts/inspect_workloads.py` (raw JSON in `experiments/workload_profile_raw.json`). Does not run the kernel._

## Summary (top 5 actionable facts)

1. **53.9% (69/128) of workloads have every batch row shorter than 2048 tokens** → `torch.topk(scores, effective_topk=min(2048, mp*64))` wastes work; per-workload K can be capped at `max(seq_lens)`. 27 workloads could use K<=1024 (21% of total), 13 could use K<=128. torch.topk is the single biggest phase (45-51 µs p50) per `profile.md`; a smaller K on 54% of workloads is a direct recoverable slice of that cost.
2. **70.9% mean early_return fraction, 94% worst-case** — the grid is severely oversized and most of it already short-circuits with exp 9's early-return. What's left: active_programs ≤ 8 for 12.5% of workloads (16/128); for those the score_kernel launch is pure dispatch overhead (B * mp launches where most return instantly) and a `grid=(sum_of_active_pages,)` flat 1-D kernel using a token-offset table would cut program count by 2-10×.
3. **96.9% of batch items have seq_len<2048** and **40.6% have seq_len<64 (one page)**. Combined with a mean intra-batch skew of 1180× (max_sl/min_sl), per-batch-item work is profoundly non-uniform. A flat grid over active (b, page) pairs instead of a padded (B, max_num_pages) rectangle eliminates padded programs at source — ~71% average grid shrinkage.
4. **Batch_size is concentrated at a handful of values**: B=1 is just 2.3% (3 workloads), B<=4 is 22.7%, B=8 is 17.2% alone, B=15 is 16.4% alone, B=29/30 is 23.4%. **B=1 fast path is NOT worth it** (3 workloads won't move the mean), but **B in {8, 15, 29, 30} covers 53 workloads (41.4%)** where a B-specialized (num_warps, BLOCK_K) might autotune well.
5. **block_table contiguity p50 = 0.95, with 32% fully contiguous rows and 54.4% of individual batch rows are a single contiguous run** — the page table is almost always a short range of consecutive pages. A contiguous-run fast path could replace the per-page `tl.load(block_table)` with a single base+offset, freeing ~1 scalar load per program (~2600 loads on the worst workload).

## Distributions

### batch_size (B)
| stat | min | p10 | p50 | p90 | max | mean |
|---|---:|---:|---:|---:|---:|---:|
| B   | 1   | 4   | 14  | 30  | 31  | 14.7 |

Cumulative: B==1: 2.3%, B<=4: 22.7%, B<=8: 45.3%, B<=16: 71.1%, B>16: 28.9%.

Most common values (counts): B=8 (22), B=15 (21), B=30 (18), B=4 (17), B=29 (12), B=2 (8).

### max_num_pages (score_kernel grid outer axis)
| stat | min | p10 | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|
| mp  | 1   | 3   | 32  | 83  | 91  |

Bimodal: heavy cluster at mp ∈ [30..45] (~50 workloads) and tail at mp ∈ [82..91] (22 workloads). 14 workloads have mp<=8.

### num_programs = B * max_num_pages
| stat | min | p10 | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|
| progs | 1 | 12 | 448 | 1335 | 2730 |

### early_return_frac (fraction of programs where `token_start >= seq_len`, killed by exp 9)
| stat | min | p10 | p50 | p90 | max | mean |
|---|---:|---:|---:|---:|---:|---:|
| frac | 0.00 | 0.31 | 0.82 | 0.90 | 0.94 | 0.71 |

### seq_lens (per batch item, n=1879 across all workloads)
| stat | min | p10 | p50 | p90 | max | mean |
|---|---:|---:|---:|---:|---:|---:|
| sl | 1 | 5 | 94 | 819 | 5806 | 313 |

Histogram: sl==1: 8.7%, sl<64: 40.6%, sl<256: 80.6%, sl<1024: 90.7%, sl<2048: 96.9%, sl>=2048: 3.1% (59 items).

### Intra-batch seq_len skew (max_sl / min_sl, for B>1, n=125)
| stat | min | p10 | p50 | p90 | max | mean |
|---|---:|---:|---:|---:|---:|---:|
| skew | 1.0 | 13.3 | 273 | 5196 | 5761 | 1181 |

| stat | min | p10 | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|
| sum_sl / (B * max_sl) | 0.05 | 0.08 | 0.17 | 0.58 | 1.00 |

Higher value means more uniform. p50=0.17 means a padded-rectangular grid wastes ~83% of per-item slots.

### Utilization = sum_sl / (B * max_num_pages * page_size)
| stat | min | p10 | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|
| util | 0.016 | 0.07 | 0.16 | 0.34 | 0.67 |

Buckets: <0.1: 21.1%, [0.1,0.3): 64.8%, [0.3,0.5): 7.8%, [0.5,0.75): 6.2%, [0.75+]: 0%.
**No workload reaches even 75% utilization.** Mean utilization 19.8%.

### effective_topk_kernel = min(2048, max_num_pages * 64)
| stat | min | p10 | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|
| ek | 64 | 192 | 2048 | 2048 | 2048 |

- 59/128 workloads have `mp*64 < 2048` (kernel already caps topk naturally).
- **69/128 workloads have ALL batch rows with sl<2048** → even where `effective_topk==2048`, the rows carry padding that `torch.topk` sorts anyway.

### max_actual_topk = max over batch of min(2048, seq_len)
| stat | min | p10 | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|
| k | 1 | 129 | 2010 | 2048 | 2048 |

k<=1024: 27 workloads; k<=512: 24; k<=256: 19; k<=128: 13; k<=64: 8.
Avg k-elements overcommitted per workload: 26.1 (small in aggregate, but concentrated on tiny workloads where a 64-way topk is 32× the work of a 2-way).

### block_table contiguity (fraction of consecutive page pairs where page[i+1] == page[i]+1)
| stat | min | p10 | p50 | p90 | max | mean |
|---|---:|---:|---:|---:|---:|---:|
| frac | 0.00 | 0.63 | 0.95 | 1.00 | 1.00 | 0.85 |

41/128 workloads fully contiguous, 73/128 >=90% contiguous. **Per-batch-row view: 1022/1879 rows (54.4%) are a single contiguous run.**

### Cross-item page reuse (total_pages_used / unique_pages_used)
| stat | min | p10 | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|
| reuse | 1.0 | 1.0 | 1.0 | 1.06 | 29.0 |

84/128 have zero reuse (factor==1.0). Only one workload shows extreme reuse (factor 29.0 = every batch item points at the same pages — the all-sl=1 case).

### block_table tail (entries beyond `num_pages_for_seq`)
- 103/128 workloads: tail values are all 0.
- 0/128: tail all -1.
- 128/128: tail values are valid page indices (0 <= v < num_pages_in_cache).
- **Consequence**: the score_kernel's exp-9 early-return on `token_start >= seq_len` is necessary. Without it, reading `block_table[b, pid_p]` for `pid_p >= num_pages_for_seq` would load page 0 (or a garbage-but-valid index), pull valid-looking FP8 data, and only the final `tl.where(in_bounds, …)` mask would suppress the write — wasted matmul.

### weights sparsity
All zeros-fraction = 0. No exploitable value sparsity in `weights`.

## Per-regime snapshot (for branch-specialized kernels)

| Regime | n | max_pg p50 | progs p50 | util p50 | early_ret p50 |
|---|---:|---:|---:|---:|---:|
| R1 B=1 tiny (sum_sl<=64) | 1 | 1 | 1 | 0.03 | 0.00 |
| R2 B=1 general | 2 | 2 | 2 | 0.51 | 0.00 |
| R3 B∈[2,4] | 26 | 8 | 16 | 0.32 | 0.43 |
| R4 B∈[5,16] | 62 | 33 | 344 | 0.16 | 0.82 |
| R5 B>=17 | 37 | 35 | 1044 | 0.12 | 0.85 |

### Workloads where score_kernel is already trivial (active_programs <= B*2 → each row uses ≤ 2 pages)
15 workloads. All-small examples (all sl<=64, B<=31):

| uuid | B | max_pg | seq_lens sample |
|---|---:|---:|---|
| 30cecff1 | 1 | 1 | [2] |
| 4667f9ad | 2 | 1 | [33, 52] |
| 46f236c0 | 4 | 1 | [19, 20, 32, 1] |
| 9410ad1e | 6 | 1 | [19, 20, 32, 12, 25, 3] |
| 752c2ee5 | 15 | 1 | [1]*15 |
| d0c00dd5 | 14 | 1 | [1]*14 |
| abc9d12c | 29 | 1 | [1]*29 |

On these, total valid tokens ≤ 64*B (fits in one CTA-sized tile). torch.topk is doing 64-way sort on ~2-64 real elements; score_kernel launches active_programs ≤ 29 tiny programs. Dispatch overhead dominates (see `profile.md`: 30cecff1 is the per-token µs outlier at 51.97 µs/tok).

## Recommended optimizations (priority ordered)

### 1. Use a per-workload `effective_topk = min(2048, max(seq_lens))` for `torch.topk`.
**What:** Replace the constant `effective_topk = min(2048, max_num_pages * page_size)` with `effective_topk_dynamic = min(2048, max_seq_lens_across_batch)`. `max_seq_lens` is a cheap `int(seq_lens.max().item())` (already paid once). The `remap_kernel` already accepts `effective_topk` as a runtime argument, so no compile-specialization change is needed. Only downside: an extra CPU↔GPU sync for `.item()`.
**Why:** `torch.topk` cost scales with `K` in the radix-select path; on 27 workloads K drops by ≥2× (2048→1024 or less), on 13 workloads K drops by ≥16×, on 8 workloads K drops to ≤64.  All-below-2048 workloads (69/128 = 53.9%) see at least *some* K reduction.
**Impact:** Roughly 50% of workloads × estimated 5-20 µs savings per call on torch.topk (exp 7 profile: topk is 45-51 µs of 104-112 µs total event time). Rough ceiling: 5-15% full-run latency reduction. Caveat: the `.item()` sync may cost ~10 µs on tiny workloads; pair it with `early_return_frac` to skip the sync when `max_num_pages*64 < 2048` already (the sync isn't needed).

### 2. Replace (B, max_num_pages) padded grid with a flat (sum_of_num_pages_for_seq) grid.
**What:** Pre-compute a pair of int32 lookup tables of length `sum(num_pages_for_seq)` on the host (cheap, CPU-only, 128-element max cost): `batch_idx_lookup[i] = b` and `pid_p_lookup[i] = p` for each active (b, p) pair. Launch `score_kernel` with grid `(sum_active,)`, load `(pid_b, pid_p)` from the lookup at program start. Drop exp 9's early-return branch (no longer needed — no empty programs).
**Why:** Mean `early_return_frac = 70.9%`, i.e. 70.9% of the current grid's programs are pure dispatch overhead (a load, compare, write of -1e30, return). For the top 10 largest workloads the active fraction is only 6-9%. `active_programs` is ≤ 256 for ALL workloads (0 workloads with active_programs > 256), while current `num_programs` reaches 2730 — that's a **~10× grid shrinkage on worst cases**. The lookup tables add 2 loads per program but remove a branch miss and a speculative (but no-op) store of `-1e30`.
**Impact:** Every workload (128/128) × estimated 1-3 µs per kernel call (score_kernel is 32-48 µs, and the savings come from dispatch overhead + fewer global-store ops for the `-1e30` sentinels). Expected 3-6% on score_kernel, larger on tiny-workload tail where dispatch is ~50% of kernel time. Side effect: eliminates the need to pre-fill scores with `-1e30` via sentinels (can instead initialize scores buffer to `-1e30` via `torch.full`, ~1 µs) or write `final_scores_masked = relu_sum * scale` only at active positions and cover the rest with an efficient large memset.

### 3. Branch-specialized small-batch fast path (B<=4 OR active_programs<=8).
**What:** A `kernel_small(...)` that skips both `torch.topk` and the `score_kernel`/`remap_kernel` pipeline when total valid tokens is tiny. Use a single Triton kernel that: (1) loads q, K from the active pages directly into registers (at most 4*64 tokens of K for B<=4 sl<=64), (2) computes scores inline, (3) sorts via `tl.sort` on the packed `(float_bits, index)` (BLOCK_N <= 512 well within the `tl.sort` budget per `LESSONS.md`). Write topk_indices directly, bypassing the scores buffer.
**Why:** 15 workloads have active_programs ≤ 8 AND max_actual_topk<=64 (sl<=64 everywhere). On these, exp 7 profile shows 104 µs total but only ~5-10 µs is real compute; **the remaining ~90 µs is py_setup + kernel launch + torch.topk overhead on nothing**. These are the worst per-token cost workloads (30cecff1 at 51.97 µs/tok is the outlier). `tl.sort` is known-good for BLOCK_N ≤ 2048 (exp 8 notes), and here BLOCK_N ≤ 512.
**Impact:** ~12% of workloads (15/128 with trivial work) × estimated 60-80 µs latency cut per call (from ~100 µs to ~30-40 µs). Mean-impact bound: 15/128 × ~50% reduction ≈ **6% on the geometric mean total run time**. The median workload will not move, but the long-tail "cheap workloads pay full price" problem disappears.

---

### Lower-priority / background notes

- **B=1 specialization not worth it** (only 3 workloads). Covered by Opt #3 anyway via `active_programs<=8`.
- **Uniform-batch specialization** (e.g. `all sl==1`, 3 workloads like 752c2ee5/d0c00dd5/abc9d12c): would collapse to a rank-1 dot product per batch item. Absorbed by Opt #3.
- **block_table-contiguity strided-load path** (32% fully contiguous, 54% of rows): theoretically one fewer load per program (replace `tl.load(block_table + pid_p)` with a compile-time stride-1 access derived from `block_table[b, 0]` + `pid_p`). The saving is ~1 scalar load per program ≈ 4-8 ns, times 2600 programs = ~10 µs on worst workloads. Worth a single A/B attempt, but the need to detect contiguity at runtime (or split into two kernels) erodes the win.
- **Page reuse (factor > 1 on 44/128)**: mostly near-1 (p90=1.06). The one extreme reuse workload (factor=29) is covered by Opt #3. Not a durable lever.
- **tail_all_zero is NOT safe to rely on**: while 103/128 workloads have tail-zero block_table, the remaining 25/128 have valid page indices in the tail (which would load real FP8 bytes). The existing exp-9 early-return is still required unless Opt #2 replaces the grid.
