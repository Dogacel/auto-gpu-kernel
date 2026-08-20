---
exp: 45
date: 2026-04-17
status: planned
parent: exp_43
---

# Experiment 45 — Flat active-only grid for score_kernel via host-precomputed (b, pid_p) lookup tables

**Hypothesis:** Replacing the padded `(batch_size, max_num_pages)` grid with a
flat `(sum_of_active_pages,)` grid plus two int32 lookup tables for `pid_b` and
`pid_p` will cut slow-path amort total by ~2-4 µs per workload on the mp>32
slow path — a measurable, mechanism-backed win that has never been tried.
Mean slow-path should drop from 46 → ~43-44 µs amort, targeting a full-run
mean Δ of -1 to -3 µs (i.e. ~-10 to -25% of the remaining full-run latency).

**Mechanism (grounded in profile.md + workload_profile.md):**

Today `score_kernel` is dispatched with `grid = (batch_size, max_num_pages)`
and exp 9 added a block-uniform early-return that fires when
`token_start >= seq_len`. `workload_profile.md` quantifies the result:
- Mean `early_return_frac = 0.71`, p90 = 0.90, max = 0.94.
- On all 128 workloads, `active_programs <= 256`, whereas `num_programs`
  reaches 2730. That's a **~10× grid shrinkage on worst cases** and
  ~3.4× mean shrinkage.

Each early-return program today still:
1. Pays launch/dispatch overhead (tens of ns in aggregate × thousands of
   programs = real wallclock on the tail workloads where SMs are oversubscribed).
2. Issues one `tl.load` of `seq_lens[b]`.
3. Issues `BLOCK_T = 64` fp32 writes of `-1e30` sentinel into `scores[b, t]`
   — i.e. `94% × 2730 × 64 × 4 B = ~660 KB` of HBM write traffic that is
   immediately overwritten by the radix kernel's own mask (`in_bounds =
   offs < max_scored`) — the bytes are dead on arrival.

By precomputing the active-(b, pid_p) list on the host (Python int loop,
executed BEFORE the kernel — no GPU sync needed because `seq_lens` is on
GPU but `block_table.shape` and the Python-side `max_num_pages` are host
values), we:
- Dispatch only the programs that will actually do work.
- Drop the sentinel-write entirely from the slow path (non-active positions
  in `scores` are initialized once via `scores.fill_(-1e30)` OR covered
  implicitly by `radix_topk_kernel`'s `in_bounds = offs < max_scored` mask
  — see "Implementation" below for the exact semantics).

The `seq_lens` tensor IS on GPU, though. We need the per-batch active
page count on the HOST to build the lookup table. This requires exactly
ONE `seq_lens.cpu()` or `seq_lens.tolist()` D→H transfer per kernel call.
Profile.md shows py_setup ~4 µs (already paid for other reasons). An int32[B]
H→D copy is ~2-4 µs for B=29 (worst case). This is the main cost risk;
see Risks below.

**Estimated recoverable time:**
- Sentinel HBM write saved: 660 KB * (1/103 GB/s) ≈ 6.4 µs on a876010b,
  but bandwidth-shared with the real score writes. Realistic recovery: ~2-3 µs.
- Dispatch overhead on the ~1900 dead programs of a876010b: ~1-2 µs.
- Downside: +2-4 µs for `seq_lens.cpu()` host round-trip, BUT only on the
  mp>32 slow path (59/128 workloads). Fast path (`max_num_pages <= 32`,
  69/128 workloads) already bypasses score_kernel entirely.
- Net: -2 to -5 µs per slow workload, -0.5 to -2 µs full-run mean.

**Implementation:**

In `kernel()` wrapper for the slow path (the `else` branch after the
`max_num_pages <= 32` short-circuit):

```python
# Host-side: compute per-batch num_active_pages = cdiv(seq_len, page_size).
# seq_lens is on GPU — pay ONE host round-trip to build the lookup table.
seq_lens_host = seq_lens.cpu().numpy()  # shape [B], int32
# Or: seq_lens.tolist() — same cost, simpler API.

# Per-batch active page count (cdiv).
num_active_per_b = [(int(sl) + page_size - 1) // page_size for sl in seq_lens_host]
# Clamp to max_num_pages (defensive; shouldn't trigger if inputs are well-formed).
num_active_per_b = [min(n, max_num_pages) for n in num_active_per_b]

total_active = sum(num_active_per_b)
# Fast-path hot short-circuit: if total_active == 0, no scoring work.
# (Won't trigger if slow path was selected — at least one batch has seq>2048.)

# Build lookup tables on host, single H→D copy.
b_lookup = []
p_lookup = []
for b, n in enumerate(num_active_per_b):
    b_lookup.extend([b] * n)
    p_lookup.extend(range(n))
b_lookup_t = torch.tensor(b_lookup, device=q_index_fp8.device, dtype=torch.int32)
p_lookup_t = torch.tensor(p_lookup, device=q_index_fp8.device, dtype=torch.int32)

# Allocate scores and pre-fill -1e30 via a dedicated fill kernel OR torch.full.
# torch.full on (B, max_scored) is 1 kernel launch, ~2 µs for 660 KB.
scores = torch.full(
    (batch_size, max_scored),
    -1e30,
    device=q_index_fp8.device,
    dtype=torch.float32,
)

# Flat grid: only active programs launched.
score_kernel_flat[(total_active,)](
    q_index_fp8, fp8_view, scale_view, weights,
    b_lookup_t, p_lookup_t,                   # NEW: per-program (pid_b, pid_p) lookup
    block_table, scores,
    # strides...
    BLOCK_H=H, BLOCK_D=D, BLOCK_T=page_size,
)
```

In `score_kernel_flat` (new kernel, mostly copy-paste of `score_kernel`):
- Load `pid_b = tl.load(b_lookup_ptr + program_id)` and
  `pid_p = tl.load(p_lookup_ptr + program_id)` instead of using 2D
  `tl.program_id`.
- **DROP** the `if token_start >= seq_len` early-return block — no longer
  needed because host-side filtering already excluded these.
- **DROP** the sentinel `tl.store(..., -1e30)` for out-of-active-range
  tiles — scores buffer is pre-filled.
- Keep the per-token `in_bounds = abs_t < seq_len` mask on the partial
  last page (within an active program, tokens past `seq_len % page_size`
  still need to be masked to `-1e30`). This is the only remaining per-token
  branch and it's already block-uniform-friendly.

`radix_topk_kernel` is UNCHANGED. Its load of `scores` uses
`mask=in_bounds, other=float('-inf')`, so positions beyond `max_scored`
are handled correctly. But positions WITHIN `max_scored` that are past
the active range now come from `torch.full(-1e30)` — equivalent to the
sentinel the old score_kernel wrote. Semantic parity preserved.

**Validation plan:**

1. **Correctness gate** — `modal run scripts/run_modal.py --quick` (2 workloads).
   Both are fast path so they won't exercise score_kernel_flat. Quick only
   catches gross structural issues. Then `--stride 8` (16 workloads, 8 slow).
   Must pass 16/16 with `matched_ratio == 1.0`.

2. **A/B gate** — `scripts/ab_benchmark.py::run --a exp_43/indexer_fused.py`.
   Paired same-VM comparison, 16 workloads. Look for:
   - Slow-path (8 workloads): mean Δ more negative than exp 43's
     ~0 ms noise-floor. Target: 4+ of 8 slow-path wins ≥ 0.0005 ms each
     (i.e. > 1 µs consistent directional signal).
   - a876010b specifically: worst-case slow workload, mp=89. Should show
     the largest absolute wall-time reduction because it has the most
     wasted programs today (~2400 dead programs per call).
   - Fast path (8 workloads): Δ ≈ 0 ± noise. The flat-grid code ONLY
     runs on `max_num_pages > 32`; fast-path workloads hit
     `fast_small_kernel` / `scoreless_kernel` and must be unaffected.
   
3. **Win criterion** — ≥4/8 slow-path wins with mean Δ ≤ -0.0005 ms on
   slow-path, OR a876010b shows ≥3% absolute reduction. Anything smaller
   is noise and the host-copy cost may be eating the gain — revert.

4. **Stretch diagnostic (only if A/B is ambiguous)**: log `total_active`
   per workload and confirm it matches workload_profile.md's
   `active_programs` (sanity on the lookup-table build). If numbers diverge
   there's a bug in the `cdiv` — fix before interpreting perf.

**Risks:**

1. **`seq_lens.cpu()` is a sync.** This is the primary risk. Per LESSONS.md
   exp 13/14, `.item()` on GPU tensors is a 30-60 µs structural barrier.
   BUT: `seq_lens.cpu()` on an int32[B] tensor (B ≤ 31) is a D→H copy,
   NOT an `.item()` — PyTorch uses `cudaMemcpyAsync` + `cudaStreamSynchronize`
   internally. The sync still gates the next kernel launch on the same
   stream. Cost should be ~3-5 µs for the full tensor (vs 30-60 µs for
   `.item()` which has per-element dispatch overhead). Detectable via A/B:
   if the host-copy cost > per-workload kernel savings, we'll see net tie
   or regression on small-mp slow workloads (mp ~ 33).
   
   **Mitigation A**: only take the sync if it's definitely amortized. Gate:
   `if max_num_pages >= THRESHOLD: use_flat_grid`. Threshold TBD but
   probably ~48 (max_num_pages ≈ 48 means B*mp > 96 programs max, early
   return frac typically still ~60%). Below that, fall back to exp 43's
   padded-grid score_kernel. One Python int compare, cheap.
   
   **Mitigation B**: pre-build lookup tables ASYNC — start the D→H on a
   side stream, do other Python work while it's in flight. Probably
   overkill for a first attempt.

2. **The `torch.full(-1e30)` is a second kernel launch.** Adds ~2 µs
   dispatch. Compared to: the current `score_kernel` writes -1e30 for
   ~71% of programs (≈1900 programs for a876010b). `torch.full` writes
   the entire scores buffer unconditionally but in one fast coalesced
   pass. For large workloads the fill is cheaper than the scattered
   sentinel stores; for small slow workloads (mp=33, B=4) it may be
   a wash or slight regression. Gate should help.
   
   **Mitigation**: pre-allocate scores buffer and reuse across calls
   (exp 16/31 ruled out Python-level pool, but this is different — the
   fill is the actual work, not an alloc-skip trick).

3. **Triton program_id → scalar load of pid_b/pid_p adds 2 HBM loads per
   program.** These are tiny (int32), should overlap with Q/K loads.
   If they serialize badly (unlikely but possible on certain schedules),
   could add 1-2 µs per slow workload. Detectable via A/B regression.

4. **Correctness corner: partial last page.** If `seq_len = 96`, then
   page 0 is full (64 tokens), page 1 has 32 valid tokens. The in-page
   mask `abs_t < seq_len` inside score_kernel_flat handles this correctly
   (positions 96..127 of page 1 get `-1e30`). BUT: in the current code,
   positions beyond the last active page (e.g. page 2+ if mp > 2) would
   have been written -1e30 by the early-return sentinel. Now they're
   covered by the `torch.full` pre-fill. Both paths converge at the
   same values — semantic parity.

**Rejected alternatives:**

- **`num_ctas=2` on score_kernel** (exp 43 "Next candidates"): thin
  ceiling, SM-bound hypothesis is wrong per profile.md (HBM-headroom,
  not SM-headroom). Risk: Blackwell CTA clustering requires grid
  divisibility + may not compile. Keep as exp 46 candidate if this
  fails.
- **Hand-rolled Gluon score_kernel with TMA + tcgen05_mma**: highest
  theoretical ceiling but (a) A/B harness broken for Gluon per exp 42
  (the `ab_benchmark.py` uses the older image without Gluon — every
  B-side workload COMPILE_ERRORs), (b) hand-rolled TMA+tcgen05_mma is
  a multi-iteration scope per exp 42 notes, and (c) block-ptr gain
  from exp 43 is absorbed (loss of 2 marginal wins). Defer to after
  we've tried the concrete-mechanism flat-grid lever.
- **num_stages sweep on score_kernel**: exp 24 closed — single `tl.dot`
  has no outer loop. block_ptr doesn't add one. No mechanism.
- **Split-H or swizzle order**: profile.md says slow-path is now
  balanced, not bottlenecked. H=64 is already small. Splitting adds
  inter-warp reduction overhead.
- **Pre-fill scores via a custom Triton kernel** instead of
  `torch.full`: the fill is launch-bound at 660 KB — `torch.full`
  should be near-optimal. Custom kernel adds one more launch
  without saving bytes.
- **Skip the pre-fill by having `radix_topk_kernel` check active
  range via a per-batch `num_active_pages` array**: possible but
  adds a second load + compare inside the radix bit loop, on the
  critical path. The pre-fill costs 2 µs once; the per-iter check
  costs more across 32 bit iterations.
- **Mitigation-B: async side-stream for lookup-table build**: added
  complexity, first test without it. If we see perf improvement but
  sync-cost limits the win, this is the follow-up.

## Coordination notes

- Quick-only will NOT exercise this change (both workloads are fast
  path). Rely on `--stride 8` (16 workloads, 8 slow) for correctness.
- A/B via `scripts/ab_benchmark.py::run --a <exp_43/indexer_fused.py>`
  is safe here — both versions are pure Triton, no Gluon-image issue.
- Implement the `max_num_pages >= 48` (or tuned) threshold gate FROM
  THE START. The whole design philosophy of LESSONS.md "host-known-shape
  branching unlocks" (exp 20) is that a single-Python-int branch is
  free; abuse it.
- If the A/B is ambiguous on mean but a876010b shows consistent -3%+,
  keep the change under a gate that requires large `max_num_pages`.
  That's still a net win structured as "this new path helps the heavy
  tail, old path unchanged for mid-slow".
- If A/B is a regression, revert immediately and propose exp 46
  `num_ctas=2` or a custom SMEM-atomic histogram radix (from exp 41
  next-candidates).
