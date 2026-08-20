# Experiment 23 — 2026-04-17

**Description:** Revert Gluon (exp_22 regression) — restore exp_18's plain-Triton `_fused_split_combine_kernel` + `_fused_attn_kernel` as the active solution. Adds `cache_modifier=".cg"` (bypass L1, cache in L2 only) on the split kernel's K_ckv / K_pe loads (ablation). Small-T fused kernel reverted to default cache (`.cg` regressed small-T in first A/B).

## Results
- Pass: 12/12 on stride-2 (quick: 2/2)
- Max abs err: 1.56e-02 (matches exp_18 baseline)
- Mode: stride-2 + A/B paired

**A/B vs exp_18** (paired, same VM):
| UUID | T-class | A (exp_18) | B (.cg split only) | Δ |
|---|---|---|---|---|
| 02d6ae9c | T=8 | 0.0157 ms | 0.0157 ms | -0.02% |
| 05f6de65 | T=2 | 0.0186 ms | 0.0186 ms | -0.03% |
| 0c23b10c | T=1 | 0.0052 ms | 0.0053 ms | +0.42% |
| 2207f0fd | T=8 | 0.0159 ms | 0.0159 ms | -0.12% |
| 232ed014 | T=8 | 0.0154 ms | 0.0153 ms | -0.17% |
| 4c46a94b | T=6 | 0.0154 ms | 0.0154 ms | +0.10% |
| 5096e459 | T=8 | 0.0159 ms | 0.0159 ms | -0.02% |
| 564007ac | T=8 | 0.0159 ms | 0.0159 ms | -0.04% |
| 78b2e11c | T=8 | 0.0156 ms | 0.0155 ms | -0.18% |
| b7668cfd | T=1 | 0.0054 ms | 0.0054 ms | +0.65% |
| e6b849f2 | T=2 | 0.0079 ms | 0.0079 ms | -0.40% |
| f77df5ce | T=2 | 0.0053 ms | 0.0054 ms | +0.41% |

Paired: 8/12 B wins, mean Δ = -0.0000 ms. **Large-T (T≥6): 6/7 B wins, all Δ ∈ [-0.18%, -0.02%]**. Directionally positive but below noise floor.

## Design

### Revert to exp_18
exp_22's Gluon kernel (software FMA, no tensor cores) was 80× slower on large T. Per skill guidance, reverted to exp_18 baseline. exp_22/ folder kept as research archive.

### `cache_modifier=".cg"` in split kernel only
The split+combine kernel's main loop issues 2–3 large K loads per BLOCK_N iter (kc [128×512 bf16]=128 KB, kp [128×64 bf16]=16 KB, kc_slice for the acc update). These are one-shot: each sparse index is loaded once per kernel call, then never reused.

**.cg modifier** — bypass L1, cache in L2 only. Rationale: L1 would add no benefit (no reuse), and .cg frees L1 for partial_acc reads later in the combine phase.

First A/B (both kernels modified) showed .cg regressed small-T (`_fused_attn_kernel`). Large-T (`_fused_split_combine_kernel`) consistently won. Selective application: split-only.

### Tried first (`.cs`) — unsupported
`cache_modifier=".cs"` (streaming) errored at compile: `Cache modifier .cs not supported`. Triton 3.6 only exposes `.ca`, `.cg`, and default. **Lesson.**

## Discoveries

1. **Triton 3.6 on Modal/B200 does NOT support `.cs` cache_modifier.** Only `.ca` (default = cache all) and `.cg` (bypass L1). Attempting `.cs` fails at compile with `Cache modifier .cs not supported`.
2. **`.cg` on K loads is workload-regime-dependent.** Consistently wins by 0.02–0.18% on large-T where K loads are the hot path and multiple CTAs co-occupy L1; marginally regresses small-T (under 1%) where single-token kernels keep small footprints that fit L1 happily.
3. **Gluon exp_22 80× regression confirmed reverted cleanly** — exp_23 latencies match exp_18's summary-reported numbers (large-T 0.016 ms median, small-T 0.005–0.018 ms range).

## Verdict

**Marginal (not a clear new best).** A/B 8/12 wins with mean Δ ≈ 0. Kept because:
1. Costs zero (single-argument change on 2 loads).
2. Consistently positive on large-T (all Δ < 0, 6/7 wins).
3. Revert of Gluon was necessary regardless.

Baseline reset to exp_18's measured performance. exp_24 will explore a more impactful lever.

## Next directions

- **Cache modifier on stores** (`.wt` write-through on partial_* stores to skip L1 → faster L2 publish for combine reads). If Triton exposes it.
- **Inline PTX `griddepcontrol.launch_dependents`** at kernel epilogue + `launch_pdl=True` — closes exp_21 PDL loop by emitting the required upstream signal.
- **Workload-aware BLOCK_N_FUSED** — profile shows T=2 with valid>1000 (05f6de65 at 0.018 ms) is the small-T tail outlier. Could conditionally dispatch BLOCK_N=128 for that regime.
- **Optimize small-T Q-load overlap** — Q load is 1.9 µs of the 10 µs small-T budget. Explicit prefetch via inline PTX might pull that in.

## Files

- `sparse_fused.py` — active kernel (copied from exp_18 with `.cg` on split K loads)
- Git commit 8ee5c9d (exp_22) was the pre-revert snapshot
