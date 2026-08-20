---
exp: 28
parent: exp_26
hypothesis: "A Triton radix-select kernel (fp32 bits, 2-pass 8-bit buckets) can replace
`torch.topk` on the slow path (mp>32). Radix-select is O(N) per pass and avoids the
`tl.sort`/`tl.argsort` BLOCK_N≥4096 wall that killed exp 8/15. Target: -20 to -30 µs on
the 59 slow-path workloads → -9 to -14 µs mean across full 128-workload set."
---

# Plan — exp 28

## Diagnosis

Current best is exp 26 at **0.0276 ms mean**. Fast path (69 workloads, mp≤32) runs ~2 µs;
**slow path (59 workloads, mp>32) stays at ~57.6 µs** because `torch.topk` is ~50% of the
event total per `profile.md` (49-61 µs p50). Exp 27 correctly diagnosed that
`score_kernel` MMA is NOT the critical-path bottleneck — GPU parallelism hides it within
one wave, so per-batch scoreless inside score_kernel saved 0 µs.

A new data point from `workload_profile_raw.json`: every slow-path workload has **exactly
one heavy batch** (seq_len ≥ 2048) plus B-1 scoreless batches (59 heavy vs 1003 light
across the slow path). `mp == ceil(max_sl/64)` on all 59 workloads — the grid width is
driven by that one batch.

Exp 10-era `stub_topk` ceiling is **38 µs** (remove torch.topk entirely). So replacing
torch.topk with a ~5-10 µs custom kernel should bring slow-path mean from 57.6 → ~30 µs:
**~-27 µs per slow-path workload × 59/128 = ~12 µs mean full-run improvement**, landing
mean at ~0.015 ms (**-45% vs exp 26**).

## Strategy

**Pivot** — implement a custom Triton **radix-select top-K** that handles
the slow path's batched top-K in a single launch, eliminating `torch.topk`'s fixed
~13 radix passes of launch overhead. Radix-select is algorithmically different from the
`tl.sort`/`tl.argsort` approaches that failed in exp 8 and exp 15 — it does NOT sort, it
partitions the value space by bit-bucket.

## Actions (priority ordered)

### 1. Implement a 2-pass radix-select Triton top-K kernel

**What:** A new `radix_topk_kernel` that replaces `torch.topk(scores, effective_topk,
dim=-1)` in the slow path. Uses fp32→monotone-uint32 bit encoding (reuse from exp 8's
correctness-verified path) and 2 passes of 11-bit histogram counting to locate the
K-th-largest score boundary, then a final compaction pass to write top-K indices.

**Algorithm (per-batch program):**

```
Inputs:  scores[max_scored] (fp32), effective_topk (int)
Outputs: topk_idx[effective_topk] (int64)

# Convert fp32 scores to monotone uint32 keys (higher key = higher score)
mono(x) = x_bits XOR ((x_bits >> 31) | 0x80000000)

# Pass 1: 11-bit histogram over top 11 bits of mono keys
#   hist1[2048] in shared memory
#   for each score: hist1[mono(score) >> 21] += 1 (masked by in-bounds)
#   reverse-cumsum hist1 → locate bucket B1 where cumsum ≥ K first
#   threshold_hi: all scores with (mono >> 21) > B1 are definitely in top-K
#   need_more: K - sum(hist1[B1+1..2047]) from bucket B1 itself

# Pass 2: 11-bit histogram over bits [10..20] within bucket B1
#   hist2[2048] in shared memory
#   for each score with (mono >> 21) == B1: hist2[(mono >> 10) & 0x7FF] += 1
#   reverse-cumsum → bucket B2 where cumsum ≥ need_more first

# Final compaction: mark as "in top-K" any score with:
#   (mono >> 21) > B1 OR ((mono >> 21) == B1 AND (mono >> 10) > B2)
#   Also include any ties in bucket (B1, B2) up to the remaining K count.
# Write indices of selected elements to topk_idx.

# Tie handling: use the low 10 bits of mono + (BLOCK_N - 1 - idx) as a
# secondary ordering key to deterministically match torch.topk's tie-break.
# For set-based matched_ratio (per LESSONS.md exp 25/26), ties don't need
# to match reference positional ordering anyway.
```

**Why radix-select over sort:**
- `tl.sort` hits a wall at BLOCK_N≥4096 (exp 8: 82→368 µs on BLOCK_N=8192; exp 15: 5×
  regression). Bitonic sort is O(N log²N) with register-pressure blowup at large tile.
- Radix-select is **O(N × num_passes)** where num_passes = 2 (with 11-bit buckets).
  Total ops per element: ~6 memory loads + 2 histogram atomics + 1 compaction compare.
- 2 passes × ~5696 elements × ~0.5 ns/element on B200 ≈ ~6 µs per batch, matching the
  `stub_topk` ceiling.
- Bucket histograms fit in 8 KB SMEM (2048 × int32), well within per-SM budget.

**Why:** torch.topk on B200 is CUB's DeviceRadixSort/RadixSelect — same algorithm class,
but pays ~13 kernel-launch-overheads due to its dispatch architecture. A single Triton
launch with fused histogramming + compaction avoids that.

**Impact:**
- Slow-path `torch.topk` phase: 49-61 µs → ~6-10 µs (~-45 µs per workload)
- Slow-path mean: 57.6 → ~15-20 µs
- Full-run mean: 0.0276 → **~0.015 ms (-45%)**
- Ceiling is set by `stub_topk` at 38 µs total, so absolute floor for slow-path is ~38 µs.
  Realistic target: 25-30 µs slow-path mean → ~0.015-0.017 ms full.

### 2. Integrate as the new torch.topk replacement (drop-in)

**What:** In `indexer_fused.py::kernel()`, after the `if max_num_pages <= 32: scoreless`
early return, replace:
```python
_, topk_idx = torch.topk(scores, effective_topk, dim=-1)  # (B, eff_topk) int64
```
with:
```python
topk_idx = torch.empty((batch_size, effective_topk), device=..., dtype=torch.int64)
radix_topk_kernel[(batch_size,)](
    scores, topk_idx,
    scores.stride(0), scores.stride(1),
    topk_idx.stride(0), topk_idx.stride(1),
    max_scored=max_scored,
    effective_topk=effective_topk,
    BLOCK_N=triton.next_power_of_2(max_scored),  # up to 8192
)
```

**Why:** Keep the downstream `remap_kernel` unchanged (it takes topk_idx as input). This
minimizes the blast radius — if radix-select is buggy, only replaces one dispatch call.

**Impact:** N/A standalone — the whole win comes from action 1's kernel.

### 3. Correctness harness first (fail-fast before perf)

**What:** Before running `/benchmark`, validate radix_topk_kernel output set-equals
`torch.topk` output on a small synthetic tensor (fp32 [4, 4096] with random values,
including some ties) via a dedicated unit test on Modal. Specifically: check
`set(our_topk_idx[b]) == set(torch_topk_idx[b])` for all b. Positional ordering doesn't
matter per LESSONS.md (set-based matched_ratio), so set-equality is the correctness bar.

**Why:** A 2-pass radix-select has 2-3 potential bugs (bucket boundary tie handling,
fp8-signed-zero monotone bit, mask-out handling for out-of-range scores that are -1e30).
Cheap to check pre-benchmark.

**Impact:** Saves an iteration if the kernel compiles but produces wrong indices.

## Do not try

- **`.item()` sync to shrink K or skip launches** (exp 13, 14): structural 60 µs barrier
  on default stream. Unavoidable even with overlap.
- **`tl.sort` / `tl.argsort` at BLOCK_N ≥ 4096** (exp 8, 15): 3-5× regression vs
  torch.topk. This is why we are using radix-select, not sort. If the radix-select
  implementation is tempted to finalize via a `tl.sort` on the narrow candidate bucket
  (bucket (B1, B2) size), cap it at BLOCK_N=256 or 512 — NOT at the full input size.
- **`tl.cat` on uint64** (exp 15): compile pass failure. Use `tl.join + tl.trans +
  tl.reshape` if concat is needed, but it's likely unneeded in a radix-select kernel.
- **Per-batch scoreless short-circuit inside score_kernel** (exp 27): tied. MMA is not
  on the critical path; don't bother eliminating it at the program level.
- **BLOCK_T=128 / PAGES_PER_PROGRAM>1 variants** (exp 3, 18, 19, 21, 22): all regressed.
  Register pressure + lost pipelining dominate any bandwidth save.
- **Aliasing DPS output as scores scratch** (exp 23): breaks torch.topk's contiguous
  fast path. Allocate scores separately as current kernel does.
- **`num_stages` / `num_warps` retuning** (exp 11, 24): no-ops on loop-free kernels.
- **Dynamic K via `seq_lens.max()`** at Python level (exp 13/14): sync barrier wins it
  back. If radix-select consumes K from a GPU-side scalar internally, that's fine, but
  don't plumb a Python-side K shrink.

## Coordination notes

1. **This is a bigger-than-usual single-iteration change** — the `radix_topk_kernel`
   itself is ~100-150 lines of Triton. Treat it as one iteration structurally but
   expect to need **exp 29 as a follow-up iteration for performance tuning** (e.g.,
   num_warps for the histogram pass, BLOCK_N specialization for workloads with small
   max_scored). If exp 28 lands correctness + modest perf win (even -5 µs mean), that's
   a structural unlock worth keeping and tuning in later iterations.

2. **Read `experiments/exp_8/indexer_fused.py` for monotone-bit encoding details** —
   this is the only prior experiment that has correctness-validated fp32→uint32 bit
   reordering code. Reusing it removes one source of bugs.

3. **Check feasibility of Triton shared memory for 2048-entry int32 histogram** before
   investing in the full implementation. If `tl.zeros([2048], tl.int32)` + atomic-add
   pattern doesn't work cleanly in current Triton, fall back to warp-aggregated
   histogram (one warp builds a private histogram, then block-level reduce).

4. **Profile-first is NOT recommended.** The slow-path kernel code is IDENTICAL to exp
   10 (no slow-path Python or Triton changes in exp 11-26); only Python dispatch
   branches above it changed (~0.3 µs overhead). `profile.md` is still structurally
   accurate for slow-path phase breakdown — torch.topk remains the dominant phase
   there.

## Fallback if the radix-select approach fails mid-implementation

If the 2-pass radix histogram doesn't compile or performs badly (>20 µs per batch):

1. **1-pass radix + compact + torch.topk on narrow candidate set**: Use a single
   11-bit histogram pass to identify the bucket containing the K-th-largest. Compact
   scores from that bucket's range into a small buffer (typically ≤ 2 × effective_topk
   elements due to bucket skew). Run a small `tl.sort` at BLOCK_N=2048 or call
   `torch.topk` on the compacted buffer. Much simpler, slightly smaller win
   (probably -15 µs instead of -27 µs on slow path).

2. **Fused score + partial-sort per page**: Each score_kernel program runs one
   64-token page, computes scores, and keeps local top-32 via `tl.sort` at BLOCK_N=64
   (proven-safe from exp 20). A second per-batch kernel merges B×mp top-32s (= up to
   ~91×32 = 2912 elements) via radix-select or a narrower `tl.sort`. Higher
   implementation complexity but reduces the top-K input by ~2× to reduce torch.topk
   cost even without eliminating it. Use only if fallback 1 also fails.

3. **Last resort — revert and profile exp 26 slow path** (Option D): if radix-select
   fundamentally doesn't land, re-run `scripts/profile_kernel_exp10.py` (adapted to
   exp 26 dispatch branches) on the three slow-path exemplars (a876010b, dba1e960,
   19e7663d). Publish updated `profile.md` to inform exp 29. This costs one iteration
   but guarantees the next one targets the right phase.
