# Experiment 50 — 2026-04-17

**Description:** Apply `eviction_policy="evict_last"` to Q_nope and Q_pe loads in `_fused_attn_kernel` (lines 230-231, the T≤2 path). Mirror of exp_48's kept variant on the combine kernel — but here the fused kernel is a 1-CTA-per-token single-kernel path (no cross-CTA L2 sharing), so the mechanism is different: L2 pinning across the 32-iter loop that walks `valid` K-tokens.

Baseline: exp_48 (Q_nope/Q_pe evict_last on `_fused_split_combine_kernel` only — combine kernel).

Hypothesis: fused kernel loops up to 32 iters per CTA; each iter loads different K/K_pe tiles (evict_first-eligible) while Q_nope/Q_pe are reused every iter — `evict_last` pins the 18 KB Q tile across the loop to reduce L2 miss rate on Q refetch.

## Results
- Pass: 128/128 (verified prior session; correctness invariant under eviction hint)
- Kernel latency: net-neutral; sub-noise across two A/B runs vs exp_48
- Mode: A/B stride 2 × 2 runs

### Stratified A/B (two runs, A=variant B=exp_48 baseline)

**T=1 workloads (0.005 ms):**
- 0c23b10c: +2.10%, +0.01% (variant faster; inconsistent magnitude)
- b7668cfd: +1.72%, +1.45% (variant faster, consistent)
- f77df5ce: +1.98%, +1.59% (variant faster, consistent)

**T=2 workload (0.008 ms):**
- e6b849f2: -0.16%, +0.77% (mixed direction)

**T≥3 combine-kernel workloads (unchanged path):**
- 7 workloads; mixed directions; magnitudes <0.5%

### Cross-session reconciliation
Prior-session summary (before context compaction) reported T=2 wins 8/8 favoring variant and T=1 loses 2/2 against variant → revert. Fresh runs show T=1 variant-favors 2/3 workloads by 1.5-2%. Direction flipped between sessions.

Absolute deltas on T=1: 0.0001 ms = 100 ns — below VM measurement precision and below cache-warming jitter. Percentages look big because denominators are tiny (5 µs). The cross-session sign flip is conclusive evidence this is pure noise.

## Learnings

Net-neutral ablation on the fused-kernel Q eviction axis. The mechanism hypothesis (pin 18 KB Q across 32-iter loop) is physically plausible — fused-kernel Q IS reused every iter — but the benefit (saving ~2-3 Q refetches × 18 KB ≈ 54 KB HBM over 32 iters) doesn't materialize measurably at 5 µs total budget.

**Cross-session noise lesson.** Marginal-keep decisions from 1-session data can flip. When a candidate sits at 100 ns absolute delta across VMs, two sessions with opposite signs = zero real signal. For future marginal calls: require BOTH (a) same-session A/B confirms direction AND (b) absolute delta exceeds ~0.0002 ms to be above the VM noise floor.

**Fused-kernel Q eviction axis closed** (net-neutral, both directions). Combine-kernel Q eviction remains kept (exp_48, 13/14 T≥3 wins across 2 runs = more robust signal from 8-way cross-CTA fan-out).

Appending LESSON-53: cross-session sign flip as the definitive noise-floor proof.

## Decision
Revert stands. Solution remains at exp_48 baseline. Continue with exp_51.
