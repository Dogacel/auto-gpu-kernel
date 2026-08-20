# Kernel Profile (exp 29 state)
_Generated 2026-04-17 against `solution/triton/indexer_fused.py` @ ac8477e (exp 29). 200 per-iter events × targets + stride-6 context (23 workloads). B200:1 via `scripts/profile_kernel_exp29.py`. Exp 10's profile preserved at `experiments/profile_exp10.md`._

## Headline

- **Cupti-equivalent (amortized event over 50-iter loop):** small_fast 18 µs, scoreless 20 µs, slow **46 µs** (uniform p50 across all 12 slow workloads, range 45.75–46.75 µs). The exp 29 bench headline of 11.6 µs full-run mean / 22.8 µs slow-path mean is on a faster VM; proportions hold across VMs.
- **Slow-path is now balanced**, not bottlenecked: `score_kernel` (17 µs GPU) ≈ `radix_topk_kernel` (16 µs GPU). py_setup + alloc CPU dispatch (~7 µs) overlaps with GPU via pipelining.
- **Event-measured totals are ~1.8× amort_total** (82-89 µs vs 46 µs) due to ~4-5 µs sync overhead per event pair × 4 pairs. Use events for **phase proportions**, amort for absolute GPU µs.

## Phase breakdown (µs, all slow path)

### Amortized (real GPU time per phase, cupti-comparable)

| Phase | small_slow (2f3b7321, mp=36) | large_slow (a876010b, mp=89) |
|---|---:|---:|
| py_setup (CPU)                | 4.4  | 4.1  |
| alloc (`torch.empty`)         | 2.8  | 2.7  |
| **score_kernel**              | **17.0** | **17.3** |
| **radix_topk_kernel**         | **14.0** | **15.8** |
| sum of GPU phases             | 33.8 | 35.8 |
| **amort_total** (end-to-end)  | **46.8** | **46.5** |
| pipeline gap (amort − sum)    | +13.0 | +10.7 |

Note: `pipeline gap` = py_setup_CPU + overhead between kernel launches. On slow-path, it's ~11-13 µs of CPU dispatch gaps that partly overlap with GPU — hence amort_total < py_setup_cpu + sum(GPU) would be expected if fully overlapped (4+36=40 µs vs 46 µs seen; ~6 µs unrecovered gap).

### Event-measured (includes sync overhead; trust only proportions)

| Phase | small_slow p50/p90 (2f3b7321) | large_slow p50/p90 (a876010b) | % large |
|---|---:|---:|---:|
| py_setup                                 | 14.4 / 14.7 | 14.2 / 14.5 | 16% |
| alloc (`torch.empty` scores)             | 10.6 / 10.9 | 10.5 / 10.8 | 12% |
| **score_kernel**                         | **30.4 / 31.5** | **37.8 / 39.0** | **43%** |
| **radix_topk_kernel**                    | **28.7 / 29.7** | **25.6 / 26.8** | **29%** |
| **TOTAL (event)**                        | 84.1 / 86.0 | 88.4 / 90.1 | 100% |
| cpu_total                                | 85.3 | 84.9 | — |

### Slow-path context (stride-6, n=12, p50)

| Phase | mp ≤ 45 (n=7) | mp ∈ [82, 91] (n=5) |
|---|---:|---:|
| py_setup   | 14.2 | 14.2 |
| alloc      | 10.5 | 10.5 |
| score      | 28.5 | 32.1 |
| radix      | 30.7 | 31.8 |
| total      | 83.8 | 88.7 |
| **amort**  | **45.9** | **46.1** |

## Observed vs memory floor

- **score_kernel byte volume (K-cache pages actually read)** for `a876010b`: sum_sl_pages × 8192 B ≈ 138 × 8192 = 1.13 MB. At 17 µs → **66 GB/s effective**. Memcpy anchor at equal byte volume: 11 µs → 103 GB/s. Memcpy is ~1.5× faster than score_kernel; score_kernel is **not memory-bound** (could get ~1.5× faster with better HBM utilization, but that's a small absolute win of ~5 µs).
- **radix_topk byte volume**: reads `B × max_scored × 4` fp32 + scatters `B × topk × 4` int32. For `a876010b`: 29 × 5696 × 4 = 661 KB in + 29 × 2048 × 4 = 238 KB out = **~900 KB**. At 16 µs → 56 GB/s effective. Floor is ~11 µs memcpy → launch + 32-bit-loop compute is ~5 µs overhead. **Radix is near memory-bound**; not worth micro-optimizing compute alone.
- Binding by regime: **slow path — both score and radix are compute-bound but near their memory floors**. Fusing would remove one HBM round-trip (~6 µs saved on scores buffer write-then-read).

## Hotspots

### Worst-5 slow-path workloads by event-measured total (p50)

| uuid | B | mp | progs | max_sl | event_total | amort | score (ev) | radix (ev) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3eab2c37 | 15 | 91 | 1365 | 5761 | 88.8 | 45.8 | 32.1 | 31.8 |
| e1a185dc | 14 | 84 | 1176 | 5321 | 88.7 | 45.9 | 31.1 | 32.8 |
| 8635db8f | 27 | 91 | 2457 | 5806 | 88.5 | 46.4 | 37.1 | 26.6 |
| **a876010b** | 29 | 89 | 2581 | 5679 | 88.4 | 46.5 | 37.8 | 25.6 |
| e4ecb462 | 15 | 82 | 1230 | 5195 | 89.1 | 46.1 | 31.5 | 32.9 |

All mp ≥ 82 (BLOCK_N=8192 for radix). The "worst" workloads are structurally identical in amortized time (~46 µs) — event-measured differences are noise/sync artifacts.

### Overlap test (isolation)

- `score_only` (a876010b, tight loop): 33.5 µs event-measured ≈ 17.3 µs amort
- `radix_only` (a876010b, tight loop): 30.8 µs event-measured ≈ 15.8 µs amort
- Back-to-back serial on default stream: 63.4 µs event-measured ≈ (17.3+15.8) = 33.1 µs amort → **no meaningful overlap**; they serialize through the scores buffer RAW dep.
- Potential save from per-batch pipelining (launch radix for batch_i as soon as score_kernel for batch_i finishes, on a second stream): ~0 µs (radix is 1 wave, starts after ALL score waves). No free lunch here.

## Current Triton config

- `score_kernel`: `BLOCK_H=64, BLOCK_D=128, BLOCK_T=64`, `num_warps=4` (default), `num_stages` unset. 2581 programs × 1 wave each on 132 SMs = ~20 waves; only 5% active due to early-return for `token_start >= seq_len` (exp 9).
- `radix_topk_kernel`: `BLOCK_N` = next_power_of_2(max_scored) = 4096 (mp ≤ 64) or 8192 (mp > 64). `num_warps=8` (exp 29 tune). Grid = `(B,)` → single wave of 14-30 programs on 132 SMs.
- `fast_small_kernel`: grid `(B,)`, used for mp==1 (8 workloads).
- `scoreless_kernel`: grid `(B, 8)`, used for mp ∈ [2, 32] (61 workloads).

Routing today: 15 mp=1 → `fast_small_kernel`; 54 mp∈[2,32] → `scoreless_kernel`; 59 mp>32 → full slow path (the 46 µs regime).

## Evaluation of exp 29's "Next candidates"

### **SUPPORTED: Fuse `score_kernel` + `radix_topk_kernel`**
The gap between `score+radix` back-to-back (46.5 µs amort) and `max(score, radix) + dispatch` (~21 µs) is ~25 µs. Eliminating the `scores` buffer HBM round-trip (write 660 KB, read 660 KB) could save ~6 µs directly; eliminating the `torch.empty` dispatch could save ~3 µs; the rest would be pipeline-gap reduction. **Estimated ceiling: 32-38 µs amort total for large slow (vs 46 µs today)** — an ~8 µs per-slow-workload save, or ~3.7 µs improvement on full-run mean across 59 workloads (~32% of total).
- Fusion constraint: `scores` is 660 KB for `a876010b` — doesn't fit in SMEM. A streaming / two-pass fusion (one program per batch, loop over mp tiles, maintain top-K as registers) is the plausible design. Tile-merge top-K at K=2048 was refuted at BLOCK_N=4096 (exp 15), but a **radix approach that streams over K tiles instead of materializing the full score tensor** is novel.
- Risk: the exp 15 lesson says tile-merge sort at K=2048 hits a B200 wall. A radix-based streaming fusion avoids that because radix sums bits rather than sorting — may work.

### **REFUTED: Early-termination of the 32-bit loop**
Radix is 16 µs amort on large. Even if every workload terminated the loop at bit 16 (half the loop), savings would be ~8 µs worst case. But this doesn't account for the non-loop portion of radix (~5 µs of scatter + load). Realistic ceiling: ~3-4 µs save per slow workload = ~1.5 µs full-run mean. Not worth pursuing; the dispatch + scatter overhead is the floor, not the bit loop.

### **REFUTED: Reducing `torch.empty` on scores (module pool, buffer reuse)**
Already tried in exp 16 and exp 31 — PyTorch's caching allocator is already doing this under the hood. Amortized alloc GPU time is 2.7 µs; event-measured 10.5 µs is dispatch overhead that overlaps with the next kernel launch. **Recoverable is ~3 µs at best**, which is what fusion would naturally include. Pursuing alloc elimination standalone is duplicative.

### **WEAK / NEEDS RE-TEST: `num_warps` tuning on `score_kernel`**
Exp 11 tried `num_warps=8` on score_kernel → regressed (15/16 workloads, 30cecff1 +12.8%). That was pre-exp-9/10 (no early-return, no scale-after-sum). Post-structural changes, the kernel body is slightly different but still single-MMA with no outer loop. The exp 29 lesson ("num_warps=8 helps reduction-heavy kernels") does NOT apply here — score_kernel has no reduction over BLOCK_N > 2048 (BLOCK_T=64, BLOCK_H=64 after cross-head sum). Ceiling: ~1-3 µs if it works. Low-value lever.

### **Per-batch scoreless inside radix**
Already implemented (line 168 in indexer_fused.py). Correct scope: the scoreless branch inside `radix_topk_kernel` handles `seq_len <= topk` batches within a larger grid. Exp 27's attempt to put the same branch inside `score_kernel` was refuted because score_kernel's wave parallelism hides per-program work.

## Bottleneck

**Phase:** The **`scores` buffer round-trip** between `score_kernel` and `radix_topk_kernel` (material through HBM) plus the dispatch gap between the two launches.
**µs:** ~8-12 µs of the 46 µs amort total (= alloc 3 µs + HBM scores round-trip ~6 µs + dispatch gap ~3 µs).
**Lever:** Fuse `score_kernel` and `radix_topk_kernel` into **one kernel with registers-resident scores per batch**. Grid: one program per batch. Inside: loop over the mp tile axis, compute scores for that tile using FP8 tensor-core dot + relu + weight + scale, and maintain a 2048-element top-K threshold/mask (radix-select on the running mask can be done once at the end over a register-backed per-batch scores vector or maintained incrementally per tile). Avoids `scores` buffer entirely.
**Ceiling if fixed:** Amort total 46 → **~32-38 µs** on slow path (~18-30% improvement). Full-run mean drops proportionally: 0.0116 → **~0.0080-0.0090 ms** (−22% to −30%).

Secondary levers (all < 3 µs full-run mean impact):
- `torch.empty` alloc elimination (duplicative with fusion)
- `num_warps` re-tune on score_kernel (exp 11 regressed; retry untested post exp 29)
- Radix bit-loop early termination (small)
- Compact grid for score_kernel (empty waves are free — no lever here)
