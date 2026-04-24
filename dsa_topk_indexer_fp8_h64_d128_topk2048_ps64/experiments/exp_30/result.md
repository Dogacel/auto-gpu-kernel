---
exp: 30
date: 2026-04-17
status: reverted
parent: exp_29
---

# Experiment 30 — 2026-04-17

**Description:** Drop `-1e30` padding from `score_kernel`. Instead of writing
`-1e30` for OOB lanes (full-tile early-return case and per-lane `abs_t >= seq_len`
mask), skip the stores entirely. Radix now masks via `offs < seq_len` so
uninitialized lanes are ignored.

## Implementation

`score_kernel`:
- Early-return branch: `if token_start >= seq_len: return` (no store).
- Active path: store raw `final` without the `in_bounds` mask.

`radix_topk_kernel`:
- `in_bounds = offs < seq_len` (was `offs < max_scored`). This mask is used by
  both the `scores` load (`other=-inf`) and the `block_table` load, and it zeros
  `mono` for OOB lanes.

All other code unchanged.

## Results

- Pass: 2/2 quick (correctness holds)
- A/B vs exp 29 (stride 8, paired, same VM):
  - B wins 5/16, mean Δ = **+0.0001 ms** → A faster (tied, slight regression)
  - Notable regressions on slow-path workloads:
    - a876010b +3.43% (0.0335 → 0.0347 ms, +1.1 µs)
    - 19e7663d +4.73% (0.0191 → 0.0200 ms)
    - 4c7705ad -2.92% (improved)
  - Fast-path workloads: within noise.
- Mode: quick + ab-vs-exp_29
- **Reverted to exp_29 state.**

## Learnings

1. **`-1e30` padding in score_kernel was a no-op on performance.** Removing the
   store for early-return tiles saves ~200 KB HBM writes on a876010b (2400
   programs × 64 × 4 bytes), which is ~0.08 µs at B200 peak bandwidth — below
   noise floor. The launch itself (seq_len load + compare + return) is the
   non-trivial cost of the early-return, not the store.
2. **Moving the correctness boundary from `max_scored` to `seq_len` didn't
   help radix either.** Both masks admit all valid tokens (since `seq_len ≤
   max_scored`). The stricter `seq_len` mask should in principle save block_table
   loads for `offs in [seq_len, max_scored)`, but this saves only ~32 loads per
   program (on avg intra-slot waste) which is negligible against the 32× `tl.sum`
   cost of the radix bit loop.
3. **Slow-path regression on two workloads suggests a compiler pessimization.**
   Removing stores sometimes lets the Triton compiler reschedule in ways that
   hurt — e.g., different register allocation, worse spill pattern, or loss of
   store-address computation that was hiding a latency. Without MLIR-level
   inspection, treat "-1e30 elimination" as risky with no upside.
4. **Code cleanup is not a performance lever here.** The -1e30 padding looked
   like cruft; it isn't.

## Next candidates

- Fuse `score_kernel` and `radix_topk_kernel`: avoids HBM round-trip but
  requires serial processing of pages per batch (mp=91 serial MMAs = slow).
- Skip `torch.empty` on scores buffer via module-level reuse (exp 16 failed on
  different grounds — worth retry post-radix).
- Compile-time `BLOCK_N` specialization (separate 4096 and 8192 variants) so
  each gets autotuned independently.
- Early-terminate radix bit loop when count matches topk exactly — requires
  dynamic loop break which Triton static_range doesn't support.
- Call profiler agent to get current breakdown (exp 10 profile is obsolete).
