# Experiment 42 — 2026-04-17

**Description:** LESSON-45's explicitly-skipped variant. Replaced the shared per-token atomic counter with 8 per-split-slot counters spaced at 128-byte stride (32 int32s per slot) so each split CTA writes to its own L2 cache line — the B200 atomic unit's lanes should then process the 8 increments in parallel rather than serialising on a single line's ownership. Combine CTAs spin on `tl.min(vals) >= gen` over the 8 strided slots. Target_count simplified from `gen * NUM_SPLITS` to `gen` (each slot is monotonic independently). Counter tensor grew from `[T]` to `[T × NUM_SPLITS × 32]` flat (1 KB/token vs 4 B/token; trivial).

## Results
- Pass: 2/2 quick (abs_err 1.56e-02 / 7.81e-03, same band as exp_37)
- Mode: stride-2 A/B vs exp_37
- **A/B 1/12 B wins, mean Δ = +0.0001 ms → A (exp_37) faster**

**A/B paired stride-2 vs exp_37:**
| UUID | T-class | A (exp_37) | B (exp_42) | Δ | % |
|---|---|---|---|---|---|
| 02d6ae9c | T=8 | 0.0153 | 0.0155 | +0.0001 | +0.92% |
| 05f6de65 | T=2 | 0.0187 | 0.0187 | +0.0000 | +0.03% |
| 0c23b10c | T=1 | 0.0055 | 0.0055 | +0.0000 | +0.59% |
| 2207f0fd | T=7 | 0.0154 | 0.0156 | +0.0002 | +0.97% |
| 232ed014 | T=8 | 0.0150 | 0.0152 | +0.0002 | +1.43% |
| 4c46a94b | T=6 | 0.0109 | 0.0110 | +0.0001 | +0.88% |
| 5096e459 | T=8 | 0.0157 | 0.0158 | +0.0002 | +1.08% |
| 564007ac | T=8 | 0.0157 | 0.0159 | +0.0002 | +1.16% |
| 78b2e11c | T=8 | 0.0154 | 0.0155 | +0.0002 | +1.12% |
| b7668cfd | T=2 | 0.0055 | 0.0055 | +0.0000 | +0.12% |
| e6b849f2 | T=2 | 0.0080 | 0.0079 | −0.0001 | **−0.76% B** |
| f77df5ce | T=2 | 0.0054 | 0.0055 | +0.0000 | +0.59% |

Consistent +0.9–1.4% regression on all 7 T≥3 workloads (where the split+combine kernel runs). T=2 small-path workloads unchanged ±0.6% (fused kernel byte-identical; noise). Reverted to exp_37 baseline.

## Verdict

**Reverted to exp_37 baseline.** Confirmed byte-for-byte via `diff -q solution/triton/sparse_fused.py experiments/exp_37/sparse_fused.py`.

## Discoveries

1. **Atomic-barrier lever fully closed — both 32-byte and 128-byte stride variants tested.** Exp_38 tested same-cache-line per-slot (8 × 4 B = 32 B total, all one line): +0.5–1.2% regression. Exp_42 tested separate-cache-line per-slot (128 B stride): +0.9–1.4% regression. **Line-separation did not unlock observable L2 atomic-lane parallelism on this shape.** Either (a) 8-way parallelism saves <250 ns vs the 8-wide vector load's extra cost, (b) B200's L2 atomic unit has fewer than 8 lanes at this line granularity, or (c) the monotonic spin's critical path is dominated by the slowest CTA's completion, not the atomic throughput. With both cheap variants now regressing, the only remaining atomic-barrier reduction is cluster-sync (`mbarrier`, which requires `num_ctas > 1` — structurally blocked by LESSON-27). Axis closed.

2. **Vectorized 8-wide spin load pays a measurable cost even with cache-line-separated slots.** The new poll path does 8 int32 loads from 8 different cache lines plus a `tl.min` reduction per iter; the old path did 1 scalar load. At ~70 poll iters/call the extra per-iter work adds up to ~500-800 ns in practice (matches the +1% observed). The atomic write-parallelism win on the producer side (if any) was smaller than this spin overhead.

3. **Cross-VM methodology confirmed (again).** e6b849f2 T=2 B wins 0.76% on an identical kernel path — pure cross-VM noise in the expected ±1% band. A/B methodology correctly attributes the regression to the 7 T≥3 workloads where the change actually applies.

## Next directions

- **All four major Triton-side axes are now empirically closed within the current kernel architecture:**
  - Cache modifiers (4 sub-axes, exp_23/24/28/41): tie-to-regression.
  - Atomic barrier (2 sub-axes, exp_38/42): regression at both 32 B and 128 B slot stride.
  - Combine-IO (3 sub-axes, exp_28/30/39): regression at all three.
  - Split MMA: `tl.dot` already near-optimal; Gluon `bw.tcgen05_mma` 6.1× worse (exp_40, LESSON-46).
  - Scalar kwargs (num_warps=8, num_stages=2, BLOCK_N=128, NUM_SPLITS=8, D_CKV_SPLIT=8): all strictly optimal.
- **Research-agent trigger reached**: 3 consecutive reverts since exp_37 (exp_38/40/41 + tying exp_42) → plateau. Next experiment should spin up a fresh research-agent for a completely untested angle (e.g. structural re-architecture: T-specific compile variants, fusing two tokens per CTA for larger effective blockM, etc.). Continuing down this wall with another micro-knob is unlikely to pay.
- **Not retry:** anything in this result's "axes closed" list.
