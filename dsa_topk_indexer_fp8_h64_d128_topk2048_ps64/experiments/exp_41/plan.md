---
exp: 41
date: 2026-04-17
status: proposed
parent: exp_37
---

# Plan — exp 41

## Diagnosis

We are at a genuine local minimum: every single-knob axis on both `radix_topk_kernel` (exps 33/34/35/36/37/38) and on the fast-path scoreless kernels (exps 25/26/40) is within the A/B noise floor, and the only structural attempt since then (exp 39 fusion) collapsed page-level parallelism and regressed +220-330%. The stale exp-29 profile still describes the live kernel's phase proportions: slow path = 17 µs score_kernel + 16 µs radix_topk + ~13 µs pipeline gap = 46 µs amort total per slow workload. Radix is near its HBM memcpy floor (~11 µs); the remaining ~5 µs of radix is the 32× `tl.sum` tree-reduction **compute** overhead of the bit-serial loop — and THAT is the only compute-bound axis in the whole pipeline that has an untried algorithmic replacement (histogram-based radix-select). This attack preserves the 2D score_kernel grid entirely (load-bearing per exp 39), is scoped to one kernel (`radix_topk_kernel`), and the per-workload ceiling is 2-4 µs on BLOCK_N=8192 slow paths — above the A/B noise floor.

## Strategy

**Targeted fix**: replace the 32-iteration bit-serial threshold-build inside `radix_topk_kernel` with a **2-pass 11-bit histogram-based radix-select**. This is the only concrete mechanism left that (a) attacks the one remaining compute-bound ~5 µs sub-phase, (b) has a real cost model from first principles (2× O(N) bucket increments + 2× 2048-bucket scan vs 32× O(N) tree reductions), and (c) doesn't touch the 2D score_kernel grid.

## Actions (priority ordered)

1. **What:** Replace the 32-iteration bit-serial loop in `radix_topk_kernel` with a two-pass 11-bit histogram radix-select over the monotone-uint32 keys. The monotone encoding + OOB masking + per-page token_idx precompute + strict/tie-prefix scatter-write are all unchanged — **we only rewrite the threshold-finding core**.

   **Pseudocode for the new core** (replaces lines 208-240 of `radix_topk_kernel` except the scatter):
   ```python
   # mono : [BLOCK_N] uint32  (as today, OOB lanes forced to 0)
   # topk : constexpr 2048
   # N_BUCKETS = 2048  (11-bit radix — compile-time)

   # ---------- Pass 1: top 11 bits (bits 31..21) ----------
   b1 = (mono >> 21) & 0x7FF          # [BLOCK_N] uint32, values in [0, 2048)
   # Build histogram in SMEM via tl.histogram (Triton primitive, built-in).
   hist1 = tl.histogram(b1, N_BUCKETS)           # [N_BUCKETS] int32
   # Reverse cumulative sum: rev_cs[i] = sum(hist[i:])
   rev_cs1 = tl.cumsum(tl.flip(hist1)) ; rev_cs1 = tl.flip(rev_cs1)  # [N_BUCKETS]
   # Find largest bucket B1 such that rev_cs1[B1] >= topk
   # i.e. the bucket containing the 2048-th largest.
   ge_mask1 = rev_cs1 >= topk
   B1 = tl.max(tl.where(ge_mask1, tl.arange(0, N_BUCKETS), 0))   # argmax of max bucket that passes
   # Strict-high: lanes with b1 > B1 are definitely in top-K.
   strict_hi_mask = b1 > B1
   strict_hi_count = tl.sum(strict_hi_mask.to(tl.int32))
   # Candidates for pass 2: b1 == B1
   cand1_mask = b1 == B1
   remaining1 = topk - strict_hi_count

   # ---------- Pass 2: low 21 bits (bits 20..0) within bucket B1 ----------
   # We need the top `remaining1` elements whose b1 == B1, ranked by their low 21 bits.
   # Take the top 11 of those 21 bits as pass-2 key (still 2048 buckets).
   b2 = (mono >> 10) & 0x7FF                      # [BLOCK_N] uint32, values in [0, 2048)
   b2_masked = tl.where(cand1_mask, b2, tl.full([BLOCK_N], 0, tl.uint32))
   # Histogram only the candidates; OOB lanes ride along at bucket 0 (hist adjusted below)
   hist2_raw = tl.histogram(b2_masked, N_BUCKETS) # [N_BUCKETS] int32
   # Subtract the fake-0 contributions from non-cand lanes:
   hist2_raw_0_adj = tl.sum((~cand1_mask).to(tl.int32))
   # Patch bucket 0 only:
   hist2 = hist2_raw - tl.where(tl.arange(0, N_BUCKETS) == 0, hist2_raw_0_adj, 0)
   rev_cs2 = tl.flip(tl.cumsum(tl.flip(hist2)))
   ge_mask2 = rev_cs2 >= remaining1
   B2 = tl.max(tl.where(ge_mask2, tl.arange(0, N_BUCKETS), 0))

   # Construct the full threshold.
   # We have chosen bits 31..21 = B1 and bits 20..10 = B2. The remaining 10 bits
   # are decided by ties in the usual strict/tie scatter. Threshold = B1<<21 | B2<<10.
   threshold = (B1 << 21) | (B2 << 10)
   # ---------- strict/tie split & scatter (reuse existing code) ----------
   strict_mask = mono > threshold
   tie_mask    = mono == threshold  # note: this is now "top-10-bits don't matter" lanes,
                                     # same scatter code as today handles it.
   # ... rest identical to current kernel ...
   ```

   **Mechanism & cost model.** Today each of the 32 `tl.sum((mono >= candidate))` calls is a tree reduction over BLOCK_N=8192 elements across 8 warps. Measured: the bit-loop accounts for ~5 µs of the 15.8 µs radix phase (profile: HBM memcpy floor is 11 µs, 16 − 11 ≈ 5 µs compute overhead). `tl.histogram` is a single HBM/SMEM pass with atomic-free accumulation (BLOCK_N threads each increment their bucket; Triton lowers to `atomicAdd` in SMEM or to warp-level histogram intrinsics). Two histogram passes over 8192 elements at ~10 GElts/s on B200 = ~1.6 µs total for both passes. Add 2× 2048-bucket `tl.cumsum` (~0.3 µs each = 0.6 µs). Net core cost: **~2.2 µs vs ~5 µs today = ~2.8 µs save per slow workload on BLOCK_N=8192**. For BLOCK_N=4096 the save is proportionally smaller (~1.5 µs).

   **Impact:** Applied to the 59 slow-path workloads × ~2 µs average save = **~0.9 µs full-run mean improvement** (11.6 → 10.7 µs). More importantly, the biggest wins land on the heaviest workloads (mp≥82, BLOCK_N=8192, 22 workloads) where ~2.8 µs × 22/128 ≈ 0.5 µs concentrated. The A/B stride-8 harness resolves ~0.0001 ms = 0.1 µs, so ~0.5-0.9 µs should be well above noise.

2. **What:** Gate the histogram path on `BLOCK_N >= 4096` (i.e. any slow-path invocation). Keep the old 32-iter bit-loop as a `tl.constexpr` branch for `BLOCK_N < 4096`. At the moment `BLOCK_N` in slow path is always 4096 or 8192 anyway, but keeping the old path makes regression-triage trivial.

   **Why:** Histogram-radix's amortized cost is driven by the fixed 2×2048-bucket scan overhead. At BLOCK_N=2048 this overhead exceeds the tree-reduction cost. Gating avoids introducing a regression on any hypothetical future shape.

   **Impact:** Compile-time branch, zero runtime cost.

3. **What:** Validate correctness carefully. Weights can be negative (exp 36 lesson), so `mono` can span the full uint32 range including the sign-flipped "negative-score" half. The 2-pass 11-bit decomposition is correct for the full range — it's just radix on uint32 keys.

   **Why:** Exp 36 (skip iter 0) failed because of a score-sign assumption. The histogram approach makes **no such assumption** — it's treating mono as a plain uint32 key. Still, test explicitly on a workload with a negative-weights batch (via correctness harness matched_ratio). Also verify the tie-path scatter still produces exactly `topk` writes — the `strict_mask | valid_tie` invariant is unchanged.

   **Impact:** Correctness gate.

## Expected outcome

- **Win case:** A/B vs exp 37 (stride 8, paired same-VM) shows B wins 6-9/16, mean Δ = -0.0005 to -0.0010 ms. Slow-path workloads mp≥82 (5-7 workloads in the stride-8 sample) at -6% to -12% each; mp∈[33,64] at -2% to -4%. No slow-path regression > +1%. Full-run mean (128 workloads): 0.0116 → 0.0105-0.0108 ms (-7% to -10%).
- **Break-even case:** `tl.histogram` on Triton / Blackwell compiles to sub-optimal SMEM atomics and its actual throughput is closer to 5 GElts/s. A/B: 8/16 wins, mean Δ ≈ -0.0001 ms. Marginal kept or reverted — diagnostic lives in the histogram throughput, not the algorithm.
- **Regression case:** Two independent failure modes:
  - **Histogram register/SMEM pressure** pushes num_warps=8 below occupancy threshold (Blackwell SMs have 100 KB SMEM; 2× 2048 × 4 B = 16 KB — well within budget). A/B wipeout across all slow-path; keep old bit-loop.
  - **tl.histogram requires int32 input** and the `uint32` cast hits a codegen path that materializes a dense mask instead of using the histogram intrinsic. A/B: +1-3% on all slow path. Mitigation: cast `mono >> 21` to int32 explicitly, or lower by hand via `tl.atomic_add` into a local buffer.

## Risks

1. **`tl.histogram` correctness on uint32**: Triton's `tl.histogram` is documented for int32 input in the standard ops. Need to cast `(mono >> 21)` to int32 (values fit in [0, 2048) so signed/unsigned is equivalent). Pre-flight: write a 20-line standalone kernel, run on Modal with quick, verify it matches a reference `torch.histogram`. If `tl.histogram` doesn't exist or doesn't compile in this Triton version, fall back to manual SMEM histogram via `tl.atomic_add(hist_ptr + b1, 1)` on a `tl.zeros([2048], tl.int32)` tile — still beats 32 tree reductions.
2. **Two-pass correctness**: the second-pass histogram masks out non-candidate lanes by mapping them to bucket 0 then subtracting their count from bucket 0. This is correct iff "bucket 0 within B1" genuinely contains some valid candidates or the subtraction leaves it non-negative. Edge case: if ALL candidates have `b2 > 0`, subtracting the non-cand count from bucket-0 gives a correct (possibly negative if non-cand > cand at bucket-0) histogram — the `rev_cs` still monotonically tracks the right count. Verified by hand on a small example (BLOCK_N=16, N_BUCKETS=4).
3. **Tie-mask semantics change**: today's tie_mask = `mono == threshold` (32-bit exact match). New tie_mask = `mono == threshold` where `threshold = (B1 << 21) | (B2 << 10)` — so ties are lanes matching the top 22 bits exactly, bottom 10 bits arbitrary. This enlarges the tie set relative to the 32-bit threshold, but the strict/tie split + `tie_prefix <= remaining` still selects exactly `topk` lanes. Invariant: `strict_count + tie_count >= topk` by construction (pass-2 ensures `rev_cs2[B2] >= remaining`). Verify via the partial-write safety check (LESSONS.md) — every slot of `topk_indices` must be written.
4. **Blackwell `tl.histogram` quality**: if histogram lowers to a serial scan (not SMEM atomics), we get 0 speedup. Pre-flight test, then decide.
5. **Correctness on workloads where effective_topk < 2048**: slow path always has `BLOCK_N >= 4096 >= topk=2048`, and the radix guarantees `count(mono >= threshold) >= topk`. Confirmed by construction.

## Why THIS approach vs other untried axes

- **Histogram radix (THIS)**: one concrete compute-bound sub-phase to attack (~5 µs), one named algorithm with a clean cost model, preserves all load-bearing parallelism. Exp 28's result.md and LESSONS.md both flagged histogram radix as the named follow-up.
- **Warp-specialized fusion**: requires Triton warp-specialization primitives (`tl.static_assert(NUM_WARPS == 8)` + manual `pid_w = tl.program_id(-1)` tricks) that are not stably supported on Blackwell Triton yet. Higher structural risk than the exp 39 attempt and a wider blast radius. Not exp-41 material — reconsider after more leverage is landed.
- **HBM prefetch hints on radix score load**: the scores load is already ~24 KB contiguous, issued at kernel start. Triton's default load already does coalesced + prefetched. Marginal (< 0.5 µs) even if it works.
- **`torch.compile` / AOT caching**: PyTorch caching allocator + Triton JIT cache already cover the warmed-up steady-state path. Launch overhead measured by profile is ~4 µs py_setup, already overlapping with GPU. Exps 13/14/16/17/31 ruled out Python-side launch levers. ~0 ceiling.
- **Different BLOCK_K on scoreless_kernel**: exp 40 closed num_warps on scoreless; BLOCK_K is another knob on same kernel. Scoreless path is ~2 µs amort — ceiling is ~1 µs × 69 / 128 ≈ 0.5 µs full-run mean. Below this proposal's ceiling.
- **Grid shape `(max_num_pages, batch_size)` instead of `(batch_size, max_num_pages)`**: reorders the SM schedule but on B200 wave scheduler picks up all 2581 programs in parallel regardless. No measurable effect expected (exp 29's profile shows score_kernel is tensor-core-utilization-limited, not scheduler-limited).
- **Seq_len-aware launch (skip empty programs at source)**: needs GPU→host sync for `max(seq_lens)` or a cumsum-based lookup table — both impose fixed costs that exceed the ~3 µs saved on the sentinel-fill. LESSONS.md `.item()` rules it out; cumsum-lookup is an exp 42+ candidate if histogram-radix lands.
- **Gluon migration**: this workload's remaining ceiling (after exp 41) is ~5-10 µs full-run mean. Gluon's marginal ceiling on top of a well-tuned Triton kernel is similar scale (maybe 2× at best on structural rewrites). Premature until radix-histogram is landed AND we're confident the 2D `(B, mp)` grid is fully saturating SMs (it is not — exp 29 profile shows 66 GB/s vs 103 GB/s memcpy, ~1.5× HBM headroom on score_kernel). The right Gluon scope would be a score-kernel rewrite, not radix — save it for exp 43+ after histogram-radix demonstrates the plateau has moved.

## Do not try

- **Skip iter 0 of bit loop** (exp 36, correctness fail — weights can be negative).
- **Scalar threshold** (exp 38, Triton already scalarizes uniforms).
- **num_warps=16 on radix** (exp 34, regressed +3-5% on BLOCK_N=4096 workloads).
- **Mask block_table by final_mask** (exp 35, HBM mask lowering is unreliable).
- **BLOCK_T=128 / two pages per program** (exp 18/19/21/22, four consecutive reverts — sort + register pressure + 2-K-page live state).
- **Fuse score_kernel + radix_topk** (exp 39, -220 to -330% regression — page-level parallelism is load-bearing).
- **Alloc elimination** (exp 16/23/31 all wash; PyTorch caching allocator already does this).
- **`.item()`-based dynamic sizing / flat-grid lookup** (exp 13/14 confirm sync is ~30-60 µs barrier).
- **`torch.as_strided` removal / SCALE_OFFSET constexpr** (exp 17 wash).
- **`num_stages` on single-dot score_kernel** (exp 24 no-op; score_kernel has no outer loop).
- **`num_warps=8` on scoreless or score_kernel** (exp 5/11/24/32/40 all tied or regressed — single-MMA/store-only kernels don't benefit from warp doubling).

## Coordination notes

- **Pre-flight**: before editing the real kernel, write a 30-line standalone Triton kernel that runs `tl.histogram` on a known input, verify compile & output. Takes ~5 min on Modal `--quick`. Saves a whole failed iteration if `tl.histogram` doesn't exist in this Triton build.
- **One structural change**: do NOT couple with num_warps / num_stages / BLOCK_N tuning. Use the exp 29 `num_warps=8`. Tune in exp 42 if it wins.
- **Benchmark plan**: `modal run scripts/run_modal.py --quick` (correctness) → `scripts/ab_benchmark.py::run --a experiments/exp_37/indexer_fused.py` paired A/B on stride 8 → if ≥5% win or directional (≥10 wins, 0 regressions >1%), run full 128.
- **Correctness checks**:
  - All 128 matched_ratio == 1.0000.
  - `topk_indices` completely overwritten (scatter invariant — see LESSONS.md "Partial-write safety"). Add a `topk_indices.fill_(-1)` temporarily as a diagnostic during first pass; remove if no partial-write bug found.
  - Include a workload with negative weights (they exist — see exp 36 lesson). Quick mode's smallest + largest workloads may not cover this; stride-8 will.
- **Read reference**: `tl.histogram` semantics — check Triton docs / source under `triton/language/standard.py` on Modal (add a `python -c "import triton.language as tl; help(tl.histogram)"` call in the pre-flight step).
- **Failure mode during implementation**: if `tl.histogram` doesn't exist, fall back to: one warp = 32 threads → give each of 8 warps a disjoint 256-bucket slice (via `pid_w = tl.warpgroup_id(); b_warp = b1 // 256`), do per-warp SMEM atomic adds into 256 buckets each, then cross-warp merge. Slightly more code but same algorithm.
