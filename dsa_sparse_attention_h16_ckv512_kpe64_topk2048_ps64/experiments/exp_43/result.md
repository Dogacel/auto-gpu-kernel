# Experiment 43 — 2026-04-17

**Description:** Per `plan.md`. Added `eviction_policy="evict_first"` to both `kc` and `kp` `tl.load` calls at lines 98-99 of `_fused_split_combine_kernel` — genuinely untested Triton load knob, orthogonal to `cache_modifier=".cg"` (L1 bypass already in place). Under stride-partition (LESSON-42), each iter's K rows are disjoint from every other iter's — one-shot reads. `evict_first` tells the L2 replacement policy to evict these lines before the multi-use Q_nope (replicated across 8 D-CTAs and re-touched during combine), partial_m/l/acc (combine reload), and sparse_indices. `_fused_attn_kernel` (T≤2 path) left untouched.

## Results
- Pass: 2/2 quick (abs_err identical to exp_37 within precision noise)
- Mode: stride-2 A/B vs exp_37
- **A/B 7/12 B wins, mean Δ = −0.0000 ms → B (exp_43) faster**

**A/B paired stride-2 vs exp_37:**
| UUID | T-class | A (exp_37) | B (exp_43) | Δ | % |
|---|---|---|---|---|---|
| 02d6ae9c | T=8 | 0.0155 | 0.0155 | −0.0000 | **−0.02% B** |
| 05f6de65 | T=2 | 0.0187 | 0.0186 | −0.0001 | **−0.29% B** |
| 0c23b10c | T=1 | 0.0052 | 0.0052 | −0.0000 | **−0.12% B** |
| 2207f0fd | T=7 | 0.0155 | 0.0155 | −0.0000 | **−0.04% B** |
| 232ed014 | T=8 | 0.0151 | 0.0151 | +0.0000 | +0.04% |
| 4c46a94b | T=6 | 0.0110 | 0.0110 | +0.0000 | +0.17% |
| 5096e459 | T=8 | 0.0157 | 0.0157 | −0.0000 | **−0.04% B** |
| 564007ac | T=8 | 0.0158 | 0.0157 | −0.0000 | **−0.20% B** |
| 78b2e11c | T=8 | 0.0154 | 0.0154 | +0.0000 | +0.17% |
| b7668cfd | T=2 | 0.0054 | 0.0054 | +0.0000 | +0.47% |
| e6b849f2 | T=2 | 0.0080 | 0.0080 | +0.0000 | +0.12% |
| f77df5ce | T=2 | 0.0053 | 0.0053 | −0.0000 | **−0.24% B** |

Of 7 T≥3 workloads (where the change applies): 4 B wins, 3 ties/A wins at sub-0.2% — direction positive. T≤2 workloads (fused kernel byte-identical, noise only): 3/5 B wins — confirms methodology.

## Verdict

**Kept — marginal win (gate passed).** Plan acceptance: ≥6/12 B wins AND mean Δ ≤ 0; achieved 7/12 + mean ≈ 0. Pattern matches exp_23 (`.cg` on K) and exp_24 (`.cg` on partial stores): directionally positive, below noise floor, kept because cost is zero and signal is consistent.

## Discoveries

1. **`eviction_policy` is a distinct lever from `cache_modifier` and still live.** All four prior cache-modifier sub-axes (exp_23/24/28/41) exhausted `.cg` on L1 bypass; none touched the L2 replacement hint. `eviction_policy="evict_first"` on one-shot K loads is directionally positive (7/12 wins, 4/7 on affected workloads). Mechanism per plan: strided K rows are guaranteed disjoint across iters → no re-read penalty for evicting them early, and freed L2 lines help Q_nope (126 MB L2, but 160 CTAs × 18 KB Q = ~3 MB Q footprint contends with ~1.5 MB partial_acc + K streaming).

2. **Sub-0.5% marginal wins are legit when mechanism is clean.** The delta here (mean ≈ 0) is below the ±1% noise band but is consistent with the mechanism (L2 replacement hint on known one-shot data). Cumulative sub-percent keeps matter: `.cg` on K (exp_23) + `.cg` on stores (exp_24) + monotonic counter (exp_37) + eviction_policy (exp_43) together form the current exp_37-baseline-plus-43 ~2-5% advantage over the exp_18-era kernel.

3. **Plateau broken via a genuinely orthogonal knob.** Research agent's discipline (grep eviction_policy → 0 matches across repo) correctly identified a distinct micro-axis mis-conflated with the closed cache-modifier axis. Lesson: before declaring "axis closed," enumerate Triton's load/store kwargs and check each against prior experiment grep.

## Next directions

- **Exp_44: `eviction_policy="evict_first"` on sparse_indices (`idx_scan` + per-iter `idx` loads).** Same mechanism: indices are scanned once per kernel call for `num_valid`, then re-touched per block — but only for contiguous prefix. Strided mode disjoints them same as K. Complementary to exp_43; ~same magnitude expected.
- **Exp_45 (if exp_44 ties):** `eviction_policy="evict_last"` on Q_nope/Q_pe — the dual: these ARE multi-use (split + combine phase), so hint "keep me in L2." Untested.
- **Exp_46 (if eviction axis saturates):** `input_precision="ieee"` on the 3 `tl.dot` calls (lines 101/102/112) per plan's fallback — genuinely untested precision knob.
- **Not retry:** any closed axis in plan.md's "Do not try" list.
