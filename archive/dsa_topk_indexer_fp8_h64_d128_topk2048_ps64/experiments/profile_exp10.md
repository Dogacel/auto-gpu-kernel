# Kernel Profile (exp 10 state, post 5 reverts)
_Generated 2026-04-17 against `solution/triton/indexer_fused.py` @ ace54ca (exp 10 kernel body; HEAD is the exp 15 revert). 100 iters on 3 target workloads + 6 context, stride 16, B200:1 via `scripts/profile_kernel_exp10.py`._

## Headline
- **cupti-measured mean** (summary.md, exp 10): **47.0 µs** (min 21 / max 75).
- **Event-measured p50 by regime**: small 93 µs, medium 104 µs, large 123 µs. Event instrumentation adds ~25 µs of sync overhead per iter — use events for **phase proportions only**, not absolute totals.
- **CPU wall-clock is flat across regimes at ~99 µs** because Python dispatch dominates the measured "total" for small workloads, not GPU work. The event-total − cpu-total gap (−26 µs on large) means event-pair syncs are themselves adding ~4 µs per pair.
- **Shape of the bottleneck**: `torch.topk` is still the single largest phase at **49-61 µs** on medium/large (47-59% of event total), followed by `score_kernel` at 26-34 µs, `py_setup` (views + shape reads) at 12-13 µs, `torch.empty` alloc at 9.4-9.8 µs, and `remap_kernel` at a flat **6.1 µs** on B≥4.

## Phase breakdown (µs, p50 / p90 by target)

| Phase | small p50/p90 (30cecff1) | medium p50/p90 (dba1e960) | large p50/p90 (a876010b) | % large |
|---|---:|---:|---:|---:|
| py_setup (shape + as_strided views)       | 12.4 / 14.5 | 12.7 / 15.4 | 12.5 / 14.8 | 10.2% |
| alloc (`torch.empty` scores buf)          |  9.4 / 10.4 |  9.6 / 11.0 |  9.8 / 10.8 |  8.0% |
| **score_kernel**                          | 25.9 / 29.5 | 27.3 / 29.8 | 33.8 / 36.7 | 27.4% |
| **torch.topk**                            | 25.6 / 27.5 | **48.9 / 50.6** | **61.2 / 61.4** | **49.7%** |
| remap_kernel                              | 19.2 / 23.1 |  6.1 /  7.1 |  6.1 /  6.1 |  5.0% |
| **TOTAL (event)**                         | 93.2 /105.2 | 104.2 /115.7 | 123.1 /130.7 | 100% |
| CPU wall-clock                            | 99.2        | 99.3        | 96.6         | —     |
| _no_alloc ceiling_ (preallocated views + buf) | 53.7   | 70.4        | 93.4         | _75.9%_ |
| _stub_topk ceiling_ (drop torch.topk)     | 38.3        | 37.8        | 38.0         | _30.9%_ |

Context (stride=16, n=6): small-regime (3 workloads) p50 total 101.8 µs, large-regime (3) p50 total 108.3 µs. Phase shares match targets within ±1 µs.

Cross-check: `no_alloc − stub_topk` attributes **15 / 33 / 55 µs to torch.topk** on small/med/large. This is a direct, events-free measurement.

## Observed vs memory floor
- **score_kernel byte volume per call**: `B·max_num_pages·64·128` FP8 bytes loaded. For a876010b (2581 programs × 8192 B = 21 MB) the Triton kernel runs in 34 µs → ~620 GB/s effective. B200 HBM is ~8 TB/s → **score_kernel has ~12× headroom in memory bandwidth** and is launch/issue-bound at small grids, compute-bound at large.
- **torch.topk byte volume**: reads `B · max_num_pages · 64 · 4` bytes (fp32 scores). For a876010b: 29 × 89 × 64 × 4 = 661 KB. At 61 µs this is only ~11 GB/s effective — torch.topk's radix-sort implementation is **launch-loop + latency bound**, not memory-bound. This is why topk saturates around 45-60 µs regardless of regime — it's paying per-pass launch overhead over ~log2(5700) ≈ 13 radix passes.
- Binding by regime: **small — dispatch overhead bound** (py_setup+alloc+remap = 41 µs of 93 µs); **medium/large — torch.topk bound** (49-61 µs of 104-123 µs).

## Hotspots

Worst by total µs (p50) among profiled workloads:

| uuid | B | pg | progs | sum_sl | tot | py | alloc | score | topk | remap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a876010b (large) | 29 | 89 | 2581 | 8812  | 123.1 | 12.5 | 9.8 | 33.8 | 61.2 | 6.1 |
| 19e7663d         | 16 | 38 |  608 | 7160  | 107.8 | 12.7 | 9.4 | 26.6 | 53.2 | 6.1 |
| bb22d09a         | 15 | 32 |  480 | 3923  | 106.4 | 12.8 | 9.5 | 26.5 | 51.8 | 6.1 |
| 2f3b7321         | 30 | 36 | 1080 | 8937  | 108.3 | 12.7 | 9.3 | 28.4 | 51.2 | 6.1 |
| dba1e960 (med)   | 25 | 30 |  750 | 10480 | 104.2 | 12.7 | 9.6 | 27.3 | 48.9 | 6.1 |

`torch.topk` is the largest single phase on every non-tiny workload. Its µs grows with **input width** (`max_num_pages · 64` = 1152 → 5696) much more than with batch. `score_kernel` grows linearly with `B·max_num_pages` (the program count) but at a shallow slope (+8 µs from 72 to 2581 programs).

## Surprises vs exp-7 profile

1. **torch.topk got *faster* on medium (48.9 vs 51 µs) and slower on large (61.2 vs 51 µs)**. The exp-7 profile was taken on a VM where `scores` alloc shared cache with a different phase; now alloc is isolated and the real topk cost on a876010b comes out higher (+10 µs).
2. **`torch.empty` alone is 9.4-9.8 µs, not 1-3 µs.** Profile 7 under-counted this — it was lumped into py_setup. Caching the scores buffer across calls could save ~9 µs flat, but is risky (dtype/shape can vary per call).
3. **`remap_kernel` is now 6.1 µs flat** on B≥4 workloads (down from 7.1 µs), and 19.2 µs on the tiny 30cecff1 (B=1, 1 remap program) — that's launch overhead, not work.
4. **score_kernel is 26-34 µs**, unchanged from exp 7 at these grid sizes. The exp-9 early-return did its job: the worst case (2581 programs, only ~294 of which have token_start < seq_len) still runs 34 µs because the inactive programs still launch + exit quickly.
5. **No "py_setup saving" from as_strided** vs exp 7's 21 µs. The new 12 µs py_setup is isolated from alloc (which we now measure separately) — combined it's 22 µs, matching exp 7's 21 µs.

## Current Triton config

`score_kernel`: `BLOCK_H=64, BLOCK_D=128, BLOCK_T=64`. Autotune **off**. `num_warps=4` (default), `num_stages` unset. exp 11 retried `num_warps=8` and regressed.

`remap_kernel`: `BLOCK_K=256`, grid `(B, 8)`. Autotune off. exp 12 retried `BLOCK_K=1024` and regressed.

## Bottleneck

**Phase:** `torch.topk(scores, 2048, dim=-1)`.
**µs:** 48.9 medium p50 / 61.2 large p50 / 47.5 context-small p50 — **~50% of event total on med/large and 25-33% of cupti-measured 47 µs headline latency.**
**Lever:** a **fused Triton top-K kernel that avoids the radix-sort tail**. Because `effective_topk=2048` and input is only `max_num_pages · 64` ≤ 5696 elements per batch on all workloads, an algorithm is available (not more radix): a **partitioned partial-sort** — per-batch pick top-2048 by first running a K=2048 priority-bound scan (register-resident threshold) across page tiles, then sorting only the survivors. exp 15's `tl.sort` attempt hit a [4096] wall; a threshold-first scan avoids sort entirely on the hot path.
**Ceiling if fixed:** the `stub_topk` measurement shows the kernel running end-to-end in **38 µs** without any topk — so **replacing torch.topk with a ~5 µs top-K gate would bring total from ~123 → ~48 µs on large** (−60%). Even a partial win (topk 61 → 25 µs) saves 36 µs on large and ~24 µs on medium.

Secondary levers:
- **alloc (9.4-9.8 µs flat)** — cacheable `scores` buffer keyed by `(B, max_num_pages)`. ~9 µs recoverable on every call.
- **py_setup (12.5 µs)** — host Python dispatch for view construction; can't eliminate but could be trimmed by merging the two `as_strided` calls into one custom layout cache.
- **score_kernel (26-34 µs)** — smallest residual lever. BLOCK_T=128 (two pages per program) would halve grid count on large; may or may not help — exp 11 showed `num_warps` retune is not free. Revisit only after top-K is solved.
- **remap_kernel (6.1 µs)** — not worth attacking.
