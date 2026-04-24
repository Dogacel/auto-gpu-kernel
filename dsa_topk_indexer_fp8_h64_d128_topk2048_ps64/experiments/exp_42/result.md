# Result — exp 42

**Verdict: REGRESSION (reverted)**

## What was tried

Ported `score_kernel` from Triton to Gluon (Triton's explicit-scheduling dialect at
`triton.experimental.gluon`), preserving all semantics from exp 37: 2D grid
`(batch, max_num_pages)`, early-return on `token_start >= seq_len`, FP8 MMA,
`relu` + per-head weight multiply, cross-head sum, scalar scale-after-sum, and
-1e30 sentinel writes. The port uses `tl_dot`, `tl_arange`, `tl_full`, `tl_trans`,
`default_blocked_layout`, `reset_to_default_layout` from
`triton.tools.triton_to_gluon_translater.translator_helpers` to wrap layout-aware
variants of the Triton language primitives.

Other kernels (`fast_small_kernel`, `scoreless_kernel`, `radix_topk_kernel`,
`kernel()` wrapper) are unchanged from exp 37.

## Preflight

- `scripts/preflight_gluon.py`: verified Gluon is importable on the Modal B200 image
  and successfully translates a reference `score_kernel` via
  `convert_triton_to_gluon` (confirmed the translator_helpers API exists).
- `scripts/preflight_gluon_score.py`: standalone correctness test against a
  PyTorch-float32 reference with synthetic inputs (B=4, H=64, D=128, page_size=64).
  Max abs err = 3.8147e-06 for all 4 batch items — well below the 0.02 numerical
  hazard threshold from CLAUDE.md.

## Benchmarks

Full-harness stride-8 correctness: **16/16 PASS, matched_ratio=1.0000** for all
trials (Gluon kernel).

**Paired A/B via `scripts/ab_benchmark.py` failed with COMPILE_ERROR on all 16
B-side workloads.** The A/B image (`nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04`
with older flashinfer-bench pin `80f40d45`) ships a Triton build that cannot
compile the Gluon port — the translator_helpers module or the Gluon experimental
package is missing/older. This is infrastructure, not a kernel bug: the regular
`run_modal.py` image (`flashinfer/flashinfer-ci-cu132:latest` + flashinfer-bench
`@main`) has Triton 3.7.0 with Gluon available and compiles + runs correctly.

Falling back to same-image cross-VM stride-8 comparison: swap exp 37 Triton into
`solution/`, run stride-8, swap Gluon back, run stride-8, compare slow-path:

### Slow-path latencies (stride-8, same `run_modal.py` image, back-to-back VM runs)

| UUID     | exp 37 Triton | exp 42 Gluon | Δ(ms) | %Δ |
|----------|--------------:|-------------:|------:|---:|
| 4c7705ad | 0.016 | 0.017 | +0.001 | +6.3% |
| 19e7663d | 0.018 | 0.020 | +0.002 | +11.1% |
| f457feb2 | 0.018 | 0.020 | +0.002 | +11.1% |
| 7f1cd9c2 | 0.018 | 0.020 | +0.002 | +11.1% |
| a876010b | 0.034 | 0.037 | +0.003 | +8.8% |
| de54c4e6 | 0.020 | 0.021 | +0.001 | +5.0% |
| 2f3b7321 | 0.021 | 0.022 | +0.001 | +4.8% |
| e63194e7 | 0.020 | 0.022 | +0.002 | +10.0% |
| **mean** | 0.0206 | 0.0224 | **+0.0017** | **+8.5%** |

Fast-path (8 workloads) is unchanged at 0.002 ms (expected — they use
`fast_small_kernel` / `scoreless_kernel`, not score_kernel). Every slow workload
slowed, consistently in the +5% to +11% range. All eight slow-path workloads
regressed.

## Why Gluon is slower

The translator_helpers helpers (`tl_dot`, `tl_trans`, `tl_arange`, `tl_full`) wrap
Gluon's layout-aware primitives with `default_blocked_layout` + `reset_to_default_layout`
round-trips around the MMA. This means the kernel pays for:

1. **Default blocked layouts that don't match the FP8 MMA-preferred layout** —
   `tl.dot` on Triton auto-picks an MMA-friendly layout; the translator falls
   back to generic BlockedLayout → NVMMADistributedLayout conversions.
2. **Explicit `convert_layout` ops** introduced by the `reset_to_default_layout`
   call around `tl.sum` and around the transposed K tile — each is a shared-memory
   transpose that Triton's autoscheduler would have folded into the dot's epilogue.
3. **No fp8 tensor core specialization** — the port compiles to `mma_v2`-style
   ops via the blocked-layout path; the Blackwell-specific `tcgen05_mma`
   pathway (with SMEM + TMEM descriptors) would require a hand-authored pipeline
   that `default_blocked_layout` doesn't emit.

Per the plan's Step 3 gate: "slow-path mean Δ ≤ +1.5 µs (within noise)". Actual
mean Δ = +1.7 µs, and % Δ = +8.5% which exceeds the ±5% parity band.

## Action

Reverted `solution/triton/indexer_fused.py` to exp 37. Kept the Gluon port in
`experiments/exp_42/indexer_fused.py` and the preflight scripts
(`scripts/preflight_gluon.py`, `scripts/preflight_gluon_score.py`) as reference
for any future Gluon retry that goes lower-level than `translator_helpers`.

## Lesson

Auto-translator Gluon (via `translator_helpers`) is not a free parity swap on
Blackwell FP8 — it's ~8-10% slower on the slow-path. A real Gluon win likely
requires hand-rolling TMA loads + tcgen05_mma + explicit SMEM staging, not just
wrapping Triton in Gluon syntax. That's a substantially bigger lift than this
iteration's scope and would need to beat the 103 GB/s memcpy ceiling referenced
in profile.md.
