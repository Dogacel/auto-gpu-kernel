# Plan — exp 51

## Hypothesis

On T=8 dispatched workloads, `split_work = 9.27 µs` (profile, 38% of kernel) is driven by a **2-iter critical path**: per-token valid ≥1032 → per-split valid ≥129 at `NUM_SPLITS=8, BLOCK_N=128` forces max_bn=256, 2 iters. LESSON-52 confirms "iter 2→3 costs ~50% of split_work". Doubling `NUM_SPLITS` to 16 halves per-split valid → drops those tokens to **1 iter**. T=8 × 16 = 128 CTAs fits on 148 B200 SMs (currently 64 → SM-underutilized).

## Mechanism

Kernel is ~1.4 µs above HBM floor (memory-bound). Per-CTA ~32 GB/s; total at 64 CTAs ~2 TB/s, well under 8 TB/s peak. **More concurrent CTAs → more concurrent HBM channels → shorter wall-clock on same total bytes.** Stride-partition (exp_26) keeps per-split valid balanced; monotonic counter (exp_37) keeps extra atomic cost flat; `s==d` coupling preserved so no kernel surgery.

NUM_SPLITS=16 under stride+monotonic is **genuinely untested**. Exp_4 was pre-stride block-partition; exp_31 tested NUM_SPLITS=4 (wrong direction). D_CKV_SPLIT=16 alone was "a wash" in exp_8; combined with doubled split parallelism on T=8, never measured.

## Change

`solution/triton/sparse_fused.py`, lines 316-318:

```python
D_CKV_SPLIT = 16    # was 8
BLOCK_D = D_ckv // D_CKV_SPLIT   # = 32
NUM_SPLITS = 16     # was 8
```

No kernel edit. Grid `(T, 16)`. Partial_acc ~8 MB — L2-resident. Combine becomes `static_range(16)` × 2-KB tiles (BLOCK_D halved); 32 KB unroll at LESSON-13's threshold, not over.

## Expected magnitude

Best: T=8 −2 to −4 µs (-13 to -25%). T=6/7 likely already 1-iter, neutral. Worst: combine overhead > split gain → ~+1 µs regression.

## Risk/fallback

Revert if: any T≥3 regresses >2% same-direction across both A/B runs; mean Δ on T=8 > 0; compile fail. Rollback = single 3-line hunk.

Falsification: T=8 split_work doesn't shrink → 2-iter premise wrong, NUM_SPLITS axis truly closed.

## Validation

LESSONS 51/52/53:

1. Quick 2/2 (abs_err ≤ 1.56e-02).
2. **Two A/B stride-2 runs vs exp_48, same VM back-to-back** (LESSON-53 noise control).
3. **Stratify** (stride-2 hits T∈{1,2,6,7,8}):
   - **T=8 (5 workloads, primary):** KEEP requires mean Δ ≤ -0.0003 ms AND ≥4/5 same-direction across both runs AND ≥1 workload ≤ -0.0005 ms absolute.
   - **T=6/T=7 (2 workloads):** tolerance ±0.0002 ms; >2% regression = don't keep.
   - **T≤2 (5 workloads):** unchanged path = noise floor baseline. If any T≤2 shifts >1%, treat as VM bias and reweight T≥3.
4. Full 128/128 only if A/B gate passes.

Do NOT cite global p50 valid=33 (LESSON-52 invalidated for stride-2). Judgment is against T=8 2-iter hypothesis only.
