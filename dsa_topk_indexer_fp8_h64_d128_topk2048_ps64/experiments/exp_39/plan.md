---
exp: 39
date: 2026-04-17
status: proposed
parent: exp_37
---

# Plan — exp 39

## Diagnosis

The last 5 experiments (34/35/36/37/38) were all micro-tunes of
`radix_topk_kernel`. Only exp 37 kept (-0.0001 ms mean A/B). We've converged
to a local minimum within the radix kernel at ~16 µs amort (profile.md).
The open lever `profile.md` explicitly flagged — **fuse `score_kernel` +
`radix_topk_kernel` to eliminate the inter-kernel dispatch gap and scores
HBM round-trip** — has never been attempted. The stale profile (exp 29)
quantifies ~8-12 µs of recoverable time per slow workload from this lever
vs the ~0.3-0.5 µs per-workload ceiling remaining in radix micro-tunes.

## Strategy

**Pivot**: move from radix micro-tuning to cross-kernel structural fusion.
The ceiling on single-kernel radix tuning is saturated; the cross-kernel
structural axis is entirely untouched and has a larger ceiling.

## Actions (priority ordered)

1. **What:** Create a single `fused_score_radix_kernel` that replaces both
   `score_kernel` + `radix_topk_kernel` on the slow path (`mp > 32`). Grid
   = `(batch_size,)`, one program per batch. Each program:
   (a) Loads Q tile `[BLOCK_H=64, BLOCK_D=128]` for this batch once.
   (b) Loops `for pid_p in range(0, cdiv(seq_len, 64))` — compute page
   score tile `[BLOCK_T=64]`, write to `scores[pid_b, pid_p*64 : (pid_p+1)*64]`
   in HBM.
   (c) After the score loop, read back `scores[pid_b, :BLOCK_N]` and run
   the existing radix-select + scatter. This is literally the body of
   `radix_topk_kernel` concatenated after the per-page score loop.

   **Why:** The per-batch program reads Q once and loops sequentially over
   pages it actually needs — identical to the current `score_kernel` grid
   collapsed along the `pid_p` axis with `pid_p`-level parallelism
   exchanged for `pid_b`-level parallelism. `profile.md` quantifies the
   recoverable gap:
   - `torch.empty` (scores) dispatch: ~3 µs → eliminated if scores alias
     a module-level scratch or is replaced by a lazy pool (exp 16 is a
     wash for isolation; in fusion context the launch gap is what
     matters, not the HBM allocation).
   - `score_kernel` launch: ~2-4 µs → absorbed into the fused launch.
   - Inter-kernel dispatch gap: ~3-5 µs → gone by construction.
   - Scores HBM write→read round-trip: ~1-2 µs (both sides hit L2, which
     is 60 MB on B200; scores for one batch = 8-23 KB, well within L2).
   - Per-batch `seq_len`-based early loop termination: the score loop
     only iterates `num_pages_for_seq = cdiv(seq_len, 64)` pages, not
     `max_num_pages`. This bakes in the workload_profile.md Opt #2
     flat-grid benefit (70.9% mean early-return saving) without
     needing a host-side sync to build a lookup table.

   **Impact:** Estimated 5-8 µs/slow-workload save × 59 slow workloads /
   128 ≈ **2.3-3.7 µs full-run mean improvement** (−20% to −32% on the
   exp 37 baseline of ~11.6 µs). Primary risk below.

   **Risk:** per-batch sequential page loop serializes the mma's within
   a batch. For a876010b (B=29, max_num_pages=89), worst-case per-program
   time is ~22 µs (89 × 0.25 µs single mma) running on 29 SMs in parallel
   vs today's 2581 parallel programs on 132 SMs (but 94% empty via exp 9
   early-return). Rough arithmetic: current score is 17 µs amort; fused
   serial could land at 15-25 µs (wider range due to single-program
   critical-path sensitivity). **The mma chain is likely HBM-load-latency
   bound for K, which pipelines well inside a single program — the
   tensor-core MMA hides behind the ~16 ns per-page K load.** Expected
   score phase amort: 12-20 µs. Even at the pessimistic end the
   dispatch/gap savings compensate.

2. **What:** Keep `scores` as an HBM buffer allocated by the caller via
   `torch.empty` (same as today). Do NOT try to eliminate `torch.empty`;
   exp 16/23/31 exhausted that axis.
   **Why:** Avoid coupling fusion with alloc-elimination. `torch.empty`
   on the PyTorch caching allocator is ~3 µs dispatch; it overlaps with
   GPU work via async execution. Isolating the fusion change keeps the
   A/B interpretable.
   **Impact:** Not a saving in itself; a constraint to keep the experiment
   bounded to ONE structural change.

3. **What:** Keep the separate `fast_small_kernel` (mp=1) and
   `scoreless_kernel` (mp≤32) paths unchanged. Fusion only replaces the
   slow path's two kernels (`score_kernel` + `radix_topk_kernel`) into
   one.
   **Why:** Fast paths are already at 2 µs amort — fusion can't help and
   changing them introduces noise. The 59 slow-path workloads are the
   only ones with score+radix round-trip cost.
   **Impact:** Preserves the 69 fast-path workload latencies within
   measurement noise.

## Do not try

- **Register-resident `[BLOCK_N]` scores tile across page iterations.**
  Triton does not support slice-write into a register tile (no
  `tile[slice] = value`). Attempted workarounds (tl.where per-position,
  64 × max_mp masked blends) are O(BLOCK_N * mp) operations that cost
  >> the HBM save. Use HBM scratch for the fusion boundary.
- **`BLOCK_T=128` (two pages per iteration).** Dead end per LESSONS.md
  after 4 tries (exp 18/19/21/22).
- **Host-side flat grid with pre-built `(b, pid_p)` lookup tables.** Would
  need `seq_lens.cpu()` (GPU→host sync, ~10-20 µs) to know which
  programs are active — wipes out the 6 µs score-kernel saving
  (LESSONS.md `.item()` sync lesson). Per-batch iteration inside the
  fused kernel naturally achieves the same "skip empty programs"
  benefit via a `seq_len`-bounded loop, no sync required.
- **Aliasing `topk_indices` as fp32 scores.** Fails on the slow path:
  min `max_scored = 33 * 64 = 2112 > 2048 = topk`, so the DPS output is
  too small to hold scores. Exp 23 also showed layout hazards with
  downstream consumers.
- **Extending the fused kernel to replace the scoreless paths.** The
  scoreless paths don't touch scores or radix — nothing to fuse.
- **Gluon / direct SMEM tensors.** Too large a departure from Triton
  for one iteration; ceiling is similar to HBM-scratch fusion (both
  save ~5-8 µs per slow workload); do this only if HBM-scratch fusion
  is confirmed to hit a floor.
- **Further radix bit-loop micro-tuning** (skip iter 0 / threshold
  shape / cumsum variants). Ceiling is < 0.5 µs per workload; below
  A/B noise floor. Five consecutive attempts already hit this floor.

## Expected outcome

- **Win case**: A/B vs exp 37 (stride 8, paired same-VM) shows B wins
  6/8 slow-path workloads at -10% to -25% each; 0 or 1 regression
  within ±2%. Full-run mean: 0.0116 → 0.0085-0.0095 ms (−20% to −30%).
  Fast-path workloads unchanged (±0.3% noise).
- **Break-even case**: mma serialization in the fused kernel's page
  loop costs exactly as much as the dispatch-gap save. A/B: 8/16 wins,
  mean Δ ≈ 0. Not kept, but informative — tells us either Gluon/SMEM
  fusion is needed or the score_kernel is already near optimal.
- **Regression case**: fused program's critical path exceeds the
  parallel score_kernel by >5 µs. Most likely cause: mma pipelining
  breaks because `tl.dot` inside a Python `for` loop compiles to
  serialized tensor-core instructions (B200's tensor cores benefit
  from cross-warp software pipelining — may need `num_stages=2` or
  manual double-buffering of K tile loads to achieve full throughput).

## Risks

1. **Single-program critical path**: one program per batch serializes
   the mma loop. Mitigations: `num_warps=8` (match radix_topk), try
   `num_stages=2` if Triton's loop pipeliner kicks in. Unlike exp 24
   (where num_stages was a no-op on single-mma score_kernel), the
   fused kernel DOES have an outer loop over pages — `num_stages`
   should pipeline K-tile loads across iterations.
2. **Register pressure**: Q tile [64, 128] fp8 + running K tile [64,
   128] fp8 + scores accumulator [64, 64] fp32 + radix state
   [BLOCK_N] uint32. At BLOCK_N=8192 this is ~40 KB of register
   state per program. At num_warps=8, 256 threads, that's 160 B/thread
   ≈ 40 fp32 regs. B200 has 255 regs/thread budget. Feasible.
3. **L2 pressure on the scores buffer**: `scores[B, max_scored]` at
   B=30, max_scored=5824 = ~700 KB. L2 is 60 MB on B200. Not a
   concern.
4. **HBM scratch aliasing hazards**: Same as today — `torch.empty`
   on the caching allocator. No new hazard.
5. **Radix correctness**: the radix phase reads scores from HBM via
   `tl.load(scores_ptr + ...)` — same pattern as today. Correctness
   preserved.

## Fallback if wins

If fusion produces a ≥5% full-run improvement (exp 39 kept):
- **exp 40**: tune `num_stages` + `num_warps` for the fused kernel (the
  outer loop over pages unlocks `num_stages` as a live axis, unlike
  exp 24). Expected additional 5-10%.
- **exp 41**: try Gluon/SMEM for the scores handoff, eliminating HBM
  entirely (only worth it if HBM traffic is the bottleneck in the
  fused kernel — profile first).
- **exp 42**: re-profile fused kernel to find the new critical path
  (likely the page-serial mma loop → potentially 2-page inner tiles
  or Q-stationary outer loop).

If fusion is break-even or regresses (exp 39 reverted):
- **exp 40**: the flat-grid approach with host sync (workload_profile
  Opt #2) deserves reconsideration — we'd know from this experiment
  that single-program serialization is too expensive, so flat grid
  across `(active_b_pid_p)` pairs with a cheap GPU cumsum replacing
  the host sync is the next natural pivot.

## Coordination notes

- **Run profile AFTER exp 39** (whether win or loss). The current
  `profile.md` is stale (exp 29). If fusion wins, the new bottleneck
  needs measurement before exp 40's tune can be directed.
- **One structural change in this iteration.** Do NOT couple with
  num_stages/num_warps tuning; use defaults first (num_warps=8 to
  match radix), tune in exp 40.
- **Benchmark plan**: `quick` (correctness first) → `stride 8` A/B
  vs exp 37 kernel snapshot → if pass, full 128. Standard procedure.
- **Correctness check priorities**:
  - All 128 matched_ratio == 1.0000.
  - `topk_indices` completely written (scatter invariant — see
    LESSONS.md "Partial-write safety").
  - Weights-negative robustness: don't perturb the radix loop's
    iter-0 semantic (exp 36 lesson). The fused kernel's radix phase
    is a byte-for-byte copy of the current `radix_topk_kernel`
    scoring branch.
