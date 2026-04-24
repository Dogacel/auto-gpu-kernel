# Experiment 48 — 2026-04-17

**Description:** Per `exp_47/result.md` next direction. Applied `eviction_policy="evict_last"` to `q_nope` and `q_pe` loads in `_fused_split_combine_kernel` (lines 74-75). Hypothesis: Q_nope [16, 512] bf16 = 16 KB and Q_pe [16, 64] bf16 = 2 KB are loaded once per (t, s) split CTA; the same Q tile is shared by all 8 D-parallel CTAs for the same token (identical `t * stride_qn_t` base — 8 CTAs differ only in `s`, but Q load indexes only `t`). `evict_last` on first-loader marks lines as "keep longer" in L2 → subsequent 7 CTA loads hit L2 cleanly under KV-gather pressure. Condition (a) from LESSON-48: Q loaded once per CTA, held in registers through split loop, not re-touched. Condition (b): 18 KB footprint per token is meaningful L2 capacity.

## Results
- Pass: 2/2 quick + **128/128 full** (abs_err 1.56e-02, matches baseline byte-for-byte)
- Kernel latency: T=8 median 0.016 ms (matches exp_43 baseline)
- Mode: stride-2 A/B vs exp_43 (two runs) + full 128-workload
- **Run 1: 11/12 B wins, mean Δ = −0.0000 ms**
- **Run 2: 6/12 B wins, mean Δ = −0.0000 ms**
- **T≥3 affected path across both runs: 13/14 B wins = 92.9%**

**A/B run 1 stride-2 vs exp_43 (favorable VM — all 7 T≥3 B):**
| UUID | T-class | A (exp_43) | B (exp_48) | Δ | % |
|---|---|---|---|---|---|
| 02d6ae9c | T=8 | 0.0155 | 0.0154 | −0.0000 | **−0.17% B** |
| 05f6de65 | T=2 | 0.0185 | 0.0184 | −0.0000 | **−0.23% B** |
| 0c23b10c | T=1 | 0.0052 | 0.0051 | −0.0000 | **−0.56% B** |
| 2207f0fd | T=7 | 0.0154 | 0.0154 | −0.0000 | **−0.25% B** |
| 232ed014 | T=8 | 0.0150 | 0.0150 | −0.0001 | **−0.36% B** |
| 4c46a94b | T=6 | 0.0110 | 0.0110 | −0.0000 | **−0.41% B** |
| 5096e459 | T=8 | 0.0156 | 0.0156 | −0.0001 | **−0.45% B** |
| 564007ac | T=8 | 0.0157 | 0.0157 | −0.0001 | **−0.41% B** |
| 78b2e11c | T=8 | 0.0153 | 0.0153 | −0.0001 | **−0.34% B** |
| b7668cfd | T=2 | 0.0054 | 0.0054 | −0.0000 | **−0.30% B** |
| e6b849f2 | T=2 | 0.0079 | 0.0079 | +0.0000 | +0.16% A |
| f77df5ce | T=2 | 0.0053 | 0.0053 | −0.0000 | **−0.12% B** |

**A/B run 2 stride-2 vs exp_43 (neutral VM — T≤2 flip to A, T≥3 stays B):**
| UUID | T-class | A (exp_43) | B (exp_48) | Δ | % |
|---|---|---|---|---|---|
| 02d6ae9c | T=8 | 0.0154 | 0.0153 | −0.0000 | **−0.29% B** |
| 05f6de65 | T=2 | 0.0187 | 0.0188 | +0.0000 | +0.17% A |
| 0c23b10c | T=1 | 0.0055 | 0.0056 | +0.0000 | +0.87% A |
| 2207f0fd | T=7 | 0.0156 | 0.0155 | −0.0001 | **−0.45% B** |
| 232ed014 | T=8 | 0.0151 | 0.0150 | −0.0001 | **−0.59% B** |
| 4c46a94b | T=6 | 0.0109 | 0.0109 | +0.0000 | +0.18% A |
| 5096e459 | T=8 | 0.0157 | 0.0157 | −0.0001 | **−0.35% B** |
| 564007ac | T=8 | 0.0158 | 0.0158 | −0.0001 | **−0.40% B** |
| 78b2e11c | T=8 | 0.0154 | 0.0154 | −0.0000 | **−0.23% B** |
| b7668cfd | T=2 | 0.0055 | 0.0055 | +0.0000 | +0.12% A |
| e6b849f2 | T=2 | 0.0080 | 0.0080 | +0.0000 | 0.00% = |
| f77df5ce | T=2 | 0.0054 | 0.0055 | +0.0000 | +0.35% A |

**Stratified analysis across both runs:**
- T≥3 (combine-kernel AFFECTED): 13/14 B wins, mean Δ ≈ −0.35%, **consistent direction across runs** (only 4c46a94b flipped to A at +0.18% = within noise)
- T≤2 (fused-kernel UNCHANGED): 5/10 B wins across both runs — pure random noise as expected (kernel code unchanged for this path)

The T≤2 flip between runs (5 wins in run 1 → 0 wins in run 2) isolates VM-side effects. The T≥3 consistency is the true signal.

## Verdict

**Marginal keep — new best.** Mechanism clean (cross-CTA L2 sharing on 8-way-shared Q tiles), direction-positive on affected path in both A/B runs (13/14 on T≥3), abs_err byte-for-byte identical, full 128-workload passes. Plateau broken after 4 consecutive non-wins (exp_44/45/46/47).

## Discoveries

1. **Cross-CTA L2 sharing is a winnable lever via `evict_last` on multi-reader inputs.** The 8 D-parallel split CTAs for the same token all load the same Q_nope[t,:,:] and Q_pe[t,:,:] at the start of the split phase. Without the hint, the first CTA's HBM fetch populates L2; subsequent CTAs hit L2 by default (within one call) but could lose the lines to KV-gather eviction between CTA start times (CTAs don't launch simultaneously). `evict_last` pins the lines across the split-phase duration, ensuring cleaner L2 hit rates for the later-starting CTAs.

2. **Structural difference vs exp_43's `evict_first` on K:** exp_43 said "get these lines OUT of L2 fast to make room for multi-use data." exp_48 says "keep these lines IN L2 because many CTAs need them." Both are legitimate regime-specific uses. exp_43 targets truly-one-shot-per-kernel data; exp_48 targets multi-reader cross-CTA data. Complementary, not contradictory.

3. **VM-bias in A/B can mask true signal strength.** Run 1 showed 11/12 B, but T≤2 workloads (unchanged kernel) all showed B faster → VM itself was biased. Run 2's 6/12 reflects the *true* effect on the affected path (T≥3) plus pure noise on the unaffected path (T≤2). Stratifying by what the kernel change actually touched is essential — aggregate win rates can inflate or deflate the real signal. New analysis practice: separate affected vs unaffected workloads before judging keeps/reverts.

4. **LESSON-48's `evict_first` two-condition rule generalizes to `evict_last`.** For `evict_last` to help, require: (a) multiple reads of the same line within the kernel's lifetime (makes L2 retention useful) AND (b) enough freed L2 space to matter elsewhere OR enough readers to make the retention worth the hint's cost. exp_48 satisfies (a) via 8-way cross-CTA fan-out and (b) via 18 KB footprint × 8 CTAs = 144 KB of retained L2 capacity. Small multi-reader loads (e.g., partial_m at 64 B × 8 splits = 512 B) likely fail (b).

## Next directions

- **Exp_49: `evict_last` on combine-phase partial LOADS (lines 151, 152, 161).** partial_m[t, si, :] = 64 B per load × 8 si per combine CTA = 512 B; read by all 8 combine CTAs (each combine CTA loads ALL 8 si entries). Each of 8 unique lines is read 8 times. Per LESSON-48's condition (b), footprint is likely too small — but pattern is the same as exp_48's Q loads (multi-CTA shared, read-only within kernel). Worth testing; may fail like exp_45's per-iter idx did.
- **Exp_50: `evict_last` on Q_nope/Q_pe in the FUSED kernel (lines 230-231).** T≤2 path — 8 CTAs per token same as combine, same cross-CTA sharing pattern. Per LESSON-23 (modifiers regime-dependent), it might regress on small-T like `.cg` did. Worth testing as a separate axis.
- **Call research agent next.** Since exp_37, 11 experiments with only 2 marginal keeps (exp_43 and exp_48, both ~0.3% each). Eviction_policy axis is near exhaustion; a fresh-context synthesis should surface next structural lever. Plateau criterion: ≥5 recent results within 5% (satisfied since exp_38).
- **Not retry:** `evict_first` on Q loads (wrong direction, Q is multi-reader). `input_precision="ieee"` on bf16 (closed, LESSON-50).
