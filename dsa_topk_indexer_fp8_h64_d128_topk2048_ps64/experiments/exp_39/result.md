---
exp: 39
date: 2026-04-17
status: reverted
parent: exp_37
---

# Experiment 39 — 2026-04-17

**Description:** Per the research agent's plan, fuse `score_kernel` +
`radix_topk_kernel` into a single `fused_score_radix_kernel` for the slow
path (`mp > 32`). Grid = `(batch_size,)`, one program per batch: load Q
once, loop over pages writing scores to HBM scratch, then run radix-select
+ scatter on the same batch. Intended to eliminate ~5-8 µs per slow
workload from kernel launch + dispatch gap + grid-scan overhead.

## Implementation

New `fused_score_radix_kernel`:
- Grid = `(batch_size,)`.
- SCORELESS branch unchanged.
- Scoring branch: Q tile loaded once, then `for pid_p in range(0, cdiv(seq_len, page_size))` with per-page MMA → HBM write.
- After loop: reload scores from HBM, run radix body identical to `radix_topk_kernel`.
- num_warps=8 (matching radix), default num_stages.

Slow-path wrapper switched from two-kernel `score_kernel` + `radix_topk_kernel`
to single-kernel `fused_score_radix_kernel`.

Tested num_stages=2 variant via `tl.range(0, n, num_stages=2)` after the
first regression.

## Results

- Pass: 2/2 quick (but quick only hits fast-path — slow path NOT exercised by quick)
- A/B vs exp_37 (stride 8):
  - Default: **B wins 3/16, mean Δ +0.0303 ms (A faster)**; all 8 slow-path workloads regressed +220-330%.
  - num_stages=2: **B wins 3/16, mean Δ +0.0339 ms** (WORSE: a876010b +388%).
- Fast-path workloads unchanged (±0.3% noise).
- **Reverted** — removed `fused_score_radix_kernel` from live file, restored `score_kernel[(batch_size, max_num_pages)]` + `radix_topk_kernel[(batch_size,)]` path.

## Learnings

**The score_kernel's 2D grid `(batch_size, max_num_pages)` is the main
source of parallelism on the slow path, not a weakness.** Collapsing to
1D `(batch_size,)` with a per-batch serial page loop is a net loss even
though it eliminates 1 kernel launch:

- For a876010b (B=29, mp=89): 29 programs × 89 serial MMAs ≈ 70 µs
  critical path per program. 29 programs on 132 SMs = sequential wall-
  time on the slowest program. Original: 2581 parallel programs on
  132 SMs = ~24-wave wall-time, hidden by tensor-core throughput.
- For other slow workloads (mp=33): 30 programs × 33 MMAs ≈ 30 µs, vs
  original's ~20 µs. Still a regression.
- **Per-MMA iteration cost is ~0.8-1.0 µs** (HBM load K + HBM load scale
  + tl.dot + reduce + HBM store scores), much higher than the research
  plan's 0.25 µs estimate. At 33-89 iterations, serialization dominates.

`num_stages=2` made it worse — doubling register pressure without
enabling enough pipelining, because the K-load + MMA + score-store
chain spans >2 stages.

## Takeaways

1. **Do not collapse a 2D grid to 1D serial loop on this problem.**
   The 94% early-return via `seq_len` filter (exp 9) from the original
   score_kernel is "wasted" work, but the high SM utilization keeps
   tensor cores fed. Serializing same work on one SM is strictly worse.
2. **Inter-kernel dispatch gap (~3-5 µs) is the price of admission**
   for SM-parallel score_kernel. Can't be eliminated by kernel fusion
   unless the fused kernel retains page-level parallelism — which
   requires either (a) keeping the 2D grid and moving the radix body
   into a third kernel, or (b) cross-warp-specialization (warps process
   different pages within one program). Both are structural rewrites
   beyond exp 39's scope.
3. **The research plan's MMA cost model was wrong.** Single-MMA cost
   on Blackwell is not 0.25 µs for fp8 64×128×128 — it's closer to
   1 µs when K-load and score-store are included. Future fusion
   proposals should benchmark a single iteration of the fused loop
   first (e.g., one-page-per-iter kernel with grid=(B,)) to validate
   the cost model before committing to N-iteration fusion.

## Next candidates

- **Retain page-level parallelism in a fused kernel**: grid = `(batch_size,
  max_num_pages)` with `pid_p`-indexed score work, but have ONE program
  per batch (e.g., `pid_p=0`) do the radix body after a `tl.atomic_xchg`
  barrier that waits for all pages. Requires inter-program sync
  primitives Triton doesn't directly expose — would need a semaphore
  pattern using `tl.atomic_add` on a counter + spin-wait. Risky.
- **Warp-specialized fusion**: single program per batch, but warps 0-N-1
  each process ⌈mp/N⌉ pages in parallel. After sync (tl.debug_barrier),
  all warps cooperate on radix. Reduces serial critical path by N=8
  (num_warps).
- **Keep exp 37, attack a different axis**:
  - scoreless path `num_warps` tuning (never done).
  - HBM prefetch via `tl.load` pipeline hints on `radix_topk_kernel`'s
    score load (currently ~24 KB at start of kernel).
  - Histogram-based radix (2-pass 11-bit) in `radix_topk_kernel` — but
    this is another radix micro-tune, below noise per exp 38.
- **Call research again** to propose a different structural direction,
  now that fusion is ruled out. Or try Gluon rewrite (long-shot).

## Experiment accounting

- Since exp 33: 34 (reverted), 35 (reverted), 36 (reverted), 37 (kept, -1%),
  38 (reverted), 39 (reverted massive). That's 5 reverts + 1 marginal win
  in 6 experiments.
- /optimize rules: after 5+ experiments at plateau, we already called
  research for exp 39. Research plan failed. Consider calling research
  with a different framing, or pivot to Gluon.
