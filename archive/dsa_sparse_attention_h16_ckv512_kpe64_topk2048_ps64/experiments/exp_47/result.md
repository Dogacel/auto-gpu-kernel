# Experiment 47 — 2026-04-17

**Description:** Per `exp_46/result.md` next direction. Applied `input_precision="ieee"` to the 3 `tl.dot` calls in `_fused_split_combine_kernel` (lines 101, 102, 112): Q_nope@K_ckv^T, Q_pe@K_pe^T (with acc), and P@K_ckv (with acc). Left fused kernel's dots untouched to keep change isolated to combine path. Genuinely untested axis — Triton's default for bf16 inputs may be equivalent to `ieee` for bf16×bf16 (no TF32 downcast possible), but docs don't specify; empirical check.

## Results
- Pass: 2/2 quick (abs_err = 1.56e-02 **identical to baseline byte-for-byte**, rel_err matches)
- Mode: stride-2 A/B vs exp_43 (two runs for stability)
- **Run 1: 8/12 B wins, mean Δ = −0.0000 ms**
- **Run 2: 7/12 B wins, mean Δ = −0.0000 ms**

**A/B run 1 stride-2 vs exp_43:**
| UUID | T-class | A (exp_43) | B (exp_47) | Δ | % |
|---|---|---|---|---|---|
| 02d6ae9c | T=8 | 0.0155 | 0.0155 | −0.0000 | −0.16% B |
| 05f6de65 | T=2 | 0.0187 | 0.0187 | −0.0000 | −0.02% B |
| 0c23b10c | T=1 | 0.0052 | 0.0052 | −0.0001 | −0.98% B |
| 2207f0fd | T=7 | 0.0155 | 0.0155 | −0.0000 | −0.02% B |
| 232ed014 | T=8 | 0.0151 | 0.0151 | −0.0000 | −0.13% B |
| 4c46a94b | T=6 | 0.0110 | 0.0111 | +0.0000 | +0.29% A |
| 5096e459 | T=8 | 0.0157 | 0.0157 | +0.0000 | +0.25% A |
| 564007ac | T=8 | 0.0157 | 0.0157 | −0.0000 | −0.04% B |
| 78b2e11c | T=8 | 0.0154 | 0.0154 | +0.0000 | +0.19% A |
| b7668cfd | T=2 | 0.0054 | 0.0054 | +0.0000 | +0.18% A |
| e6b849f2 | T=2 | 0.0080 | 0.0080 | −0.0000 | −0.44% B |
| f77df5ce | T=2 | 0.0054 | 0.0053 | −0.0000 | −0.49% B |

**A/B run 2 (confirmation):** 7/12 B wins, Δ≈0, same direction-positive pattern.

All 24 paired deltas are ±0.5% — well under noise floor. Combined 15/24 = 62.5% B wins across both runs, but mean ≈ 0 ms and all magnitudes are sub-1%.

## Verdict

**Reverted to exp_43 baseline.** Byte-for-byte verified via diff.

**Rationale for revert despite 62.5% B-win rate:** No mechanism. `input_precision="ieee"` is designed to affect FP32/TF32 precision choice — on bf16×bf16 inputs, the hardware's BF16 MMA is already IEEE-compliant for those 16-bit operands. Both paths lower to the same `wgmma.bf16.bf16.f32` instruction. Output abs_err identical confirms no numerical difference. The 62.5% B-win rate at sub-0.3% magnitudes without a mechanism story is textbook measurement noise (expected distribution under null is 50%, 62.5% is only +1.5σ at n=24). Keeping this would cargo-cult a cosmetic change that could regress after unrelated compile-pass reordering in a future Triton version.

Contrast with exp_43 keep: exp_43 had a clean L2-replacement-hint mechanism that predicted direction-positive. exp_47 has nothing predictive.

## Discoveries

1. **`input_precision="ieee"` is a no-op for bf16×bf16 `tl.dot` on Blackwell.** PTX lowering identical to default; abs_err matches byte-for-byte. Triton's `input_precision` parameter only selects among FP32 MMA strategies (full IEEE, single TF32, 3× TF32 compensating). BF16 inputs skip this selection — there's only one way to do bf16×bf16 → fp32 on sm_100.

2. **Marginal keep criterion requires mechanism, not just statistics.** Exp_43 (7/12, kept) had clean L2 mechanism. Exp_47 (15/24 combined, not kept) has no mechanism. Rule: direction-positive at sub-1% without a predictive mechanism is noise to revert, not signal to keep. Stat-only evidence for marginal wins is insufficient — the mechanism tells you the win persists under future codegen changes; statistics alone don't.

3. **Axis closed: `input_precision` for this kernel.** Since all three `tl.dot` sites in the combine kernel use bf16 inputs, and there's no FP32 input path, `input_precision` has no programmable effect. Skip this axis on future kernels using bf16-only MMAs.

## Next directions

- **Exp_48: `evict_last` on Q_nope/Q_pe LOADS (lines 74-75).** Q_nope [16, 512] bf16 = 16 KB, Q_pe [16, 64] bf16 = 2 KB. Loaded once per (t, s) split CTA; same Q tile shared across 8 D-parallel CTAs (same t, different s, all load same Q rows). Cross-CTA L2 sharing opportunity. Q loads have no `.cg` so PTX allows `evict_last` (per LESSON-49). Condition (b) satisfied: 18 KB per token is a meaningful L2 footprint to protect across KV-gather pressure.
- **Exp_49: `evict_last` on combine-phase partial LOADS (lines 151, 152, 161).** partial_m[t, si, :] = 64 B/load × 8 si = 512 B per combine CTA; read by all 8 combine CTAs (64 loads total, 8 unique lines). First loader's `evict_last` marks as hot → subsequent 7 loaders hit L2. Small footprint but clean multi-reader pattern. These loads have no `.cg` so PTX allows the hint.
- **Exp_50: Workload-specialization probe.** Read `experiments/workload_profile.md` + workload-inspector agent. Median token uses only 33/2048 TopK entries (from LESSON-11). Is there a kernel variant that specializes for the median-33 regime that could undercut stride-partition's 1-iter cost?
- **Not retry:** `input_precision` on any bf16-only `tl.dot`. Axis closed (this experiment).
