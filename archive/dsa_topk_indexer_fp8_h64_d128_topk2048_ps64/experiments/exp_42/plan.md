# Plan — exp 42

## Diagnosis

Since exp 33 (last real win), 8 experiments: 1 marginal (exp 37, block_table hoist, -0.0001 ms
A/B), 7 reverts, including 2 catastrophic regressions (exp 39 fusion +220-330%, exp 41
histogram-radix +327-443%). Every single-knob axis on `radix_topk_kernel` is now closed
(34/35/36/38 micro-axes, 41 algorithmic) and so is `num_warps` on every other kernel
(5/11/24/32/40). Per `profile.md`, slow-path amort is 46 µs = score(17) + radix(16) +
gap(13); **score_kernel runs at 66 GB/s vs 103 GB/s memcpy ceiling — ~1.5× HBM headroom
is the only remaining above-noise ceiling on the whole pipeline**. Radix is already near
memcpy floor (16 µs − 11 µs = ~5 µs compute left, below A/B noise floor per exp 33-38).

## Strategy

**Pivot — Gluon migration of `score_kernel` (the one compute-bound phase with real
HBM headroom)**. Hand-scheduled CUTLASS-style load/MMA/epilogue pipeline via Gluon.
Every Triton micro-lever is closed; `tl.histogram` demonstrated that Triton 3.7's codegen
can miss by 30× on less-standard patterns; further radix micro-tuning has ceiling < 0.25 µs
full-run mean (below the ~0.0001 ms A/B floor).

Execute via a **fresh-context sub-agent** per `/optimize` Gluon-migration rule — the agent
re-reads CLAUDE.md and writes a Gluon port of `score_kernel` only, keeping `scoreless_kernel`,
`fast_small_kernel`, and `radix_topk_kernel` unchanged. **Scope: score_kernel only. Radix and
fast paths are untouched this iteration.**

## Actions (priority ordered)

1. **What:** Spin up a fresh-context sub-agent via the `Agent` tool to write a Gluon version
   of `score_kernel` only. The agent should: (a) read `CLAUDE.md` + `solution/triton/
   indexer_fused.py` + `experiments/profile.md` for bottleneck context, (b) produce
   `solution/triton/indexer_fused.py` with a `score_kernel` implemented in Gluon,
   preserving the current 2D grid `(batch_size, max_num_pages)`, early-return on
   `token_start >= seq_len`, fp8 MMA, relu + weight multiply, cross-head sum, scalar
   scale-after-sum, and -1e30 sentinel writes. (c) keep the Python wrapper identical,
   (d) preserve DPS output contract.
   **Why:** score_kernel is the *only* remaining kernel with a measurable HBM headroom
   above noise (66 → 103 GB/s = 1.5× ceiling from `profile.md`). Triton's scheduler
   apparently doesn't reach the memcpy ceiling for single-MMA kernels on Blackwell —
   exp 37 already showed manual load hoisting recovers latency the compiler misses.
   Gluon's explicit load/MMA pipeline gives direct control.
   **Impact (grounded in profile.md, not first-principles):** slow-path score_kernel at
   17 µs amort × 59/128 workloads. Memcpy floor on the same byte volume is 11 µs. Even
   capturing 50% of the 6 µs headroom → ~3 µs slow-path save → **~1.4 µs full-run mean
   improvement** (from ~11.6 µs to ~10.2 µs, -12%). Full 1.5× (if unicorn): ~2.8 µs
   full-run mean, -24%. Ceiling dwarfs radix micro-axes (< 0.25 µs). Realistic: -10% to
   -15% full-run mean.

2. **What:** Constrain the Gluon sub-agent to do **nothing but port** score_kernel in
   iteration 1 — no fusion with radix, no 2D-to-1D grid change, no extra tiles. The
   point is to establish Gluon-parity first, then tune via subsequent iterations.
   **Why:** Exp 39 catastrophically regressed when fusion+grid-change coupled with per-MMA
   cost estimation errors. Decoupling Gluon port from structural change isolates the
   mechanism. If Gluon-parity lands within ±5% of Triton score_kernel, the migration is
   viable and subsequent iterations can tune (load pipelining depth, warp-specialized
   prefetch, async copy with `cp.async`-like primitives in Gluon).
   **Impact:** No new best expected in exp 42 itself — the A/B target is "within ±5% of
   exp 37" on slow-path. Exp 43+ is where Gluon-specific wins would land. This matches
   the `/optimize` Gluon-migration guidance ("not guaranteed to be faster than triton").

3. **What:** A/B gate. Benchmark via `modal run scripts/ab_benchmark.py::run
   --a solution/triton/indexer_fused.py` vs `experiments/exp_37/indexer_fused.py`
   (current live best). Keep only if: (a) correctness 16/16 matched_ratio=1.0,
   (b) slow-path mean Δ ≤ +1.5 µs (within noise + compile-schedule variance),
   (c) fast-path and scoreless paths unchanged (they use different kernels and
   shouldn't shift).
   **Why:** Gluon parity means "we have a compiled Gluon kernel we can tune from". If
   it's slower by >5% with no obvious fix, it's a failed port; revert and reconsider.
   **Impact:** Sets the floor for exp 43's Gluon-tuning work.

## Cost model (grounded in profile.md)

- score_kernel amort = 17 µs (profile.md, exp 29 state, unchanged post-exp-37)
- memcpy floor for same byte volume (1.13 MB on a876010b at 103 GB/s) = 11 µs
- Headroom = 17 − 11 = **6 µs per slow workload**, 59 slow workloads
- Capture 50% → 59 × 3 / 128 / 1000 = **1.4 µs full-run mean** = **-12% from 11.6 µs**
- Capture 100% (unicorn, unlikely first Gluon iter) → 2.8 µs / -24%
- Baseline (no-tune Gluon parity): 0 µs improvement (success = "didn't regress")

## Do not try

- **Another Triton micro-tune on radix** — ceiling < 0.25 µs full-run mean (exp 33-38
  confirmed below A/B noise).
- **Manual SMEM histogram for radix** — exp 41 was 30× off; ceiling for radix is bounded
  at ~5 µs per slow workload = 2.3 µs full-run; the hand-roll risk surface (atomic
  contention on 2048 buckets × 256 threads) has similar failure modes to `tl.histogram`
  itself. Revisit only if Gluon itself closes off.
- **Fusion of score + radix** — exp 39 catastrophic; per-MMA cost underestimated 3-4×.
  Any future fusion proposal requires a single-iter stub benchmark first (exp 39 lesson).
- **Warp-specialized fusion** — exp 40 research note flagged Triton warp-spec primitives
  as unstable on 3.7; combined with the 1D-grid collapse risk from exp 39, this is high-
  probability regression.
- **Dynamic grid from block_table** — ceiling ~0.9 µs full-run mean (sentinel-fill save
  on 59 slow workloads × ~2 µs), at A/B noise floor. Low ROI vs Gluon's 1.4 µs ceiling.
- **`num_warps` tuning on any existing kernel** — score (5/11/24/32), scoreless (40),
  fast_small, radix (29 kept, 34 reverted). All explored.
- **`.item()` / CPU-sync-based K shrinking** — exp 13/14 rule out sync as structural
  barrier.
- **BLOCK_T > 64 on score_kernel** — exp 18/19/21/22 all reverted.

## Coordination notes

- **Fresh-context sub-agent required.** Per `/optimize` Gluon-migration section:
  "spin up a new sub-agent with a fresh context". Do NOT inline the migration into
  the current optimizer context — keep this plan as the sole bridge, and let the
  Gluon agent work from CLAUDE.md + indexer_fused.py + profile.md.
- **Scope discipline**: score_kernel only. No fusion, no grid change, no radix
  touches. Iteration 1 establishes Gluon viability.
- **Stride-8 A/B is the decision metric** (per CLAUDE.md "sub-5% deltas vs previous
  best → ab_benchmark.py"). Cross-VM summary.md numbers are noise.
- **If exp 42 passes parity**, exp 43 is Gluon-specific tuning (async loads,
  load/MMA pipelining, warp partitioning within the program). If exp 42 regresses
  >5%, revert and fall back to (c.2) manual SMEM histogram as a final Triton
  hail-mary, or (c.3) re-call research with "Gluon attempted, regressed" framing.
- **Correctness check**: compile (2 workloads quick) + A/B stride 8 (16 workloads
  paired same-VM). No full 128 until parity confirmed.
