# Experiment 8 — 2026-04-17

**Description:** Attempted to replace `torch.topk(scores, 2048, dim=-1)`
with a Triton `topk_kernel` using `tl.sort` on a packed `(monotone_float32_bits,
index)` int64 key. One program per batch, `BLOCK_N = next_pow2(max_scored)`.

## Result: REVERTED — net neutral/regression

- Pass: 128/128 correctness (exact match via monotone-bit encoding)
- Full run mean: 0.0509 ms vs exp 7's 0.049 ms → **+3.8% regression**
- A/B vs exp 7 (stride 8): B wins 6/16, mean Δ = +0.0000 ms (tied)

## Phase-by-phase behavior (stride 8)

Attempted two variants:

### V1: Unconditional triton topk_kernel (BLOCK_N = next_pow2(max_scored))
- Small workloads: 30cecff1 22 µs → 11 µs (**−50%**)
- Medium (BLOCK_N=4096): ~+27% regression (112 → 143 µs)
- Large (BLOCK_N=8192, a876010b): **82 µs → 368 µs (+349%)**

### V2: Branch `if BLOCK_N <= 2048: triton else: torch.topk`
- Small: preserved (11-12 µs)
- Medium: no change from exp 7 in theory, but A/B shows +15% on some
  (likely VM noise + small Python-branch overhead)
- Large: preserved (no regression)
- **Net: mean Δ = +0.0000 ms (tied)**. Full run: 0.0509 vs 0.049 ms = +3.8%.

## Why tl.sort scales badly past BLOCK_N=2048

`tl.sort` in this Triton version (installed on Modal B200) appears to use
a bitonic/merge sort. For BLOCK_N=8192, the kernel runs ~290 µs longer
than torch.topk (the radix-select implementation) on the same input size.
Per-program cost analysis: 1024 threads × 169 compare-swap phases should
compute in ~5-10 µs given B200's compute throughput. The observed 280 µs
overhead suggests `tl.sort` on uint64 keys isn't fully vectorized on this
hardware, or hits register-pressure/SMEM spills at BLOCK_N=8192.

## Correctness (positive finding)

Monotone float32 → uint32 conversion + packing with index works
for elementwise-exact match:
- Packing: `mono_bits = f32_bits ^ ((f32_bits >> 31) | 0x80000000)` (via uint32 ops)
- Packed key: `(mono.u64 << 32) | (BLOCK_N - 1 - idx).u64`
- Sort descending by packed key → exact match with torch.topk tie-breaking.

matched_ratio = 1.0 on all 128 workloads in both V1 and V2.

## Verdict

Reverted to exp 7 kernel. Triton `tl.sort` is only a win when
BLOCK_N ≤ 2048, but the small-workload wins are below noise threshold
in the full-run mean (0.001 ms absolute improvement on 27 small
workloads gets washed out by 101 medium/large at Δ ≈ 0).

## Next directions

1. **Smarter top-K than `tl.sort`**: radix-select (2-3 pass histograms)
   stays O(N) instead of O(N log²N). Could match torch.topk up to
   BLOCK_N=8192 and win on smaller.
2. **Attack score_kernel next** (32-48 µs, 33%). Autotune num_warps and
   try larger BLOCK_T with care (LESSONS.md warns BLOCK_T>64 regresses
   under current tile shape, but BLOCK_D/BLOCK_H variants unexplored).
3. **Reduce py_setup (21 µs, 19%)**. Pure Python overhead — fewer
   `.stride()` calls, pre-captured constexprs, single Python path.
4. **Fuse score + top-K**: each score_kernel program keeps top-K_local
   per page (e.g., top-32 out of 64) via a bitonic sort of its 64
   scores. Second kernel merges per-batch. Worst case doesn't save
   bandwidth (see exp 8 notes) but the top-K input to the merge is
   smaller, so a Triton top-K over 2848 elements is more tractable.
