# Plan — exp 43

## Diagnosis
Five consecutive reverts/ties since exp_37 (exp_38, 39, 40, 41, 42) on four
independent axes (atomic-barrier slot stride, combine-IO merge, Gluon MMA,
`.cg` on Q). Per `profile.md`, the kernel is within ~1 µs CUPTI of its
estimated Triton ceiling (14.5 µs floor vs 15.5 µs current). One load-side
micro-knob has never been tested per `grep eviction_policy`:
`tl.load(..., eviction_policy=...)`. It is orthogonal to
`cache_modifier=".cg"` (the sub-axis exhausted exp_23/24/28/41) — `.cg`
bypasses L1, `eviction_policy` steers L2 replacement.

## Strategy
**targeted fixes.** One-line addition of `eviction_policy="evict_first"` on
the `kc`/`kp` stride-partition loads inside `_fused_split_combine_kernel`.
Mechanism: under stride-partition each CTA's `BLOCK_N=128` K rows per iter
are disjoint from every other iter's rows (LESSON-42) — a true one-shot
read. `evict_first` frees L2 capacity for the lines that *do* reuse:
Q_nope (replicated across 8 D-split CTAs, re-touched in combine),
partial_m/l/acc (combine-phase reload), and sparse_indices.

## Actions (priority ordered)
1. **What:** `sparse_fused.py` lines 98–99 — add
   `eviction_policy="evict_first"` to both `kc = tl.load(...)` and
   `kp = tl.load(...)` in the `for bn in range(0, max_bn, BLOCK_N)` loop of
   `_fused_split_combine_kernel`. Leave `_fused_attn_kernel` (T≤2)
   untouched — its kc/kp tiles are small and hot in L2.
   **Why:** K/V gather tiles are one-shot under stride-partition; keeping
   them in L2 post-consumption wastes capacity Q_nope and partial_m/l/acc
   actually use. `.cg` already bypasses L1; `evict_first` is the
   complementary L2 hint and has never been tested.
   **Impact:** Plausible +0.2–0.5% on T≥3 if L2 eviction contributes to
   Q-reload or combine-load latency. Realistic worst case: neutral
   (±0.2%).

## Gate & fallback
- **Acceptance (stride-2 A/B vs exp_37):** B wins ≥ 6/12 AND mean Δ ≤ 0.
  Expected magnitude is sub-0.5% → tie-or-better criterion, aligned with
  kept exp_23/24.
- **Fallback (exp_44):** If ties/regresses, exp_44 probes
  `input_precision="ieee"` on the 3 `tl.dot` calls at lines 101/102/112
  (genuinely untested per `grep`; cited as "next direction" in exp_1 but
  never ran). If that ties, pursue T∈{6,7}-specific compile variant — but
  that requires decoupling the NUM_SPLITS == D_CKV_SPLIT assertion (line
  347), too large for one iteration.

## Do not try
- Any `cache_modifier` on Q/K/partial loads or stores (LESSON-40/46;
  exp_23, 24, 28, 41).
- Per-slot atomic barriers at any stride (LESSON-45; exp_38, 42).
- Partial_m/l merge, scratch persistence, `.cg` on combine loads
  (exp_28, 30, 39).
- Gluon `bw.tcgen05_mma` or any MMA-based compute path at H=16 (LESSON-46;
  exp_22, 40).
- `num_warps`, `num_stages`, `BLOCK_N`, `NUM_SPLITS`, `D_CKV_SPLIT`,
  `D_CKV_SPLIT_FUSED` sweeps on either kernel (exp_14/17/19/32/33/34/36).
- `num_ctas` / cluster with atomic barrier (LESSON-27).
- `launch_pdl=True` with or without upstream `griddepcontrol.launch_dependents`
  (LESSON-29/41; exp_21, 25 — invisible to CUPTI).
- Compact-block partition (exp_27, 29); buffer persistence for
  partial_m/l/acc (exp_30).

## Why this hasn't been tried
`cache_modifier` and `eviction_policy` are **distinct Triton load knobs**
commonly conflated. All four closed cache-modifier sub-axes tested `.cg`
placement (L1 bypass); none touched `eviction_policy` (L2 replacement hint:
`evict_first` / `evict_last` / `evict_normal`). `grep eviction_policy`
over `experiments/` and `solution/` returns zero matches. The mechanism is
also qualitatively different from LESSON-40: that concerns per-iter
tag-check cost (L1), whereas this targets L2 capacity contention between
the one-shot KV stream and multi-use Q/partial state. Stride-partition
(LESSON-42) makes this cleaner than it would have been under exp_18's
block-partition: rows are guaranteed disjoint, so `evict_first` carries no
hidden re-read penalty.

## Coordination notes
Single-line probe; stride-2 A/B vs exp_37 is sufficient. No profile rerun
(probe magnitude is below the ~1 µs VM noise floor). If keeps, commit; if
ties, revert via `cp experiments/exp_37/sparse_fused.py solution/triton/`
and log as exp_43.
