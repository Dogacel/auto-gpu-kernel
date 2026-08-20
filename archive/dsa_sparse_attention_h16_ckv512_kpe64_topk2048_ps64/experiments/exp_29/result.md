# Experiment 29 — 2026-04-17

**Description:** Retry compact-block partition (exp_27's concept) with explicit `in_range = pos < end_s` bounds mask on the idx load. Exp_27's correctness bug was diagnosed: when `range_s` isn't divisible by `BLOCK_N`, the last loop iteration overshoots `end_s`; since those positions are still in the valid prefix (`idx >= 0` true), they double-count into multiple splits' softmax → wrong output. Fix: mask on idx load + `valid = (idx >= 0) & in_range`.

## Results
- Pass: **2/2 quick** (correctness fix worked)
- Mode: quick + A/B vs exp_26 (stride-2)
- **Reverted** — large regression across all large-T

**A/B run vs exp_26 (stride-2):**
| UUID | A (exp_26) | B (exp_29) | Δ | % |
|---|---|---|---|---|
| 02d6ae9c | 0.0160 | 0.0176 | +0.0016 | +10.04% |
| 05f6de65 | 0.0188 | 0.0188 | +0.0000 | +0.26% |
| 0c23b10c | 0.0053 | 0.0053 | −0.0000 | −0.00% |
| 2207f0fd | 0.0159 | 0.0176 | +0.0017 | +10.75% |
| 232ed014 | 0.0155 | 0.0170 | +0.0015 | +9.70% |
| 4c46a94b | 0.0115 | 0.0127 | +0.0012 | +10.52% |
| 5096e459 | 0.0162 | 0.0179 | +0.0016 | +9.93% |
| 564007ac | 0.0163 | 0.0179 | +0.0016 | +9.96% |
| 78b2e11c | 0.0160 | 0.0175 | +0.0015 | +9.41% |
| b7668cfd | 0.0054 | 0.0055 | +0.0000 | +0.76% |
| e6b849f2 | 0.0081 | 0.0080 | −0.0001 | −1.07% |
| f77df5ce | 0.0054 | 0.0054 | +0.0000 | +0.30% |

Paired: A wins 10/12, mean Δ = +0.0009 ms (B 6% slower overall, 10% on large-T).

## Design (reverted)

```python
offs_topk = tl.arange(0, SPLIT_SIZE * NUM_SPLITS)  # = TOPK = 2048
idx_scan_all = tl.load(Indices_ptr + t * stride_idx_t + offs_topk)
num_valid_total = tl.sum((idx_scan_all >= 0).to(tl.int32), axis=0)
start_s = (s * num_valid_total) // NUM_SPLITS
end_s = ((s + 1) * num_valid_total) // NUM_SPLITS
range_s = end_s - start_s
max_bn = ((range_s + BLOCK_N - 1) // BLOCK_N) * BLOCK_N

for bn in range(0, max_bn, BLOCK_N):
    pos = start_s + bn + offs_n
    in_range = pos < end_s
    idx = tl.load(Indices_ptr + t * stride_idx_t + pos, mask=in_range, other=-1)
    valid = (idx >= 0) & in_range
    safe_idx = tl.where(valid, idx, 0).to(tl.int64)
    # rest unchanged
```

## Discoveries

1. **Correctness bug diagnosed and fixed.** The missing mask at the last iteration was the root cause of exp_27's T=8 `abs_err=2.82`. With `in_range` mask both quick workloads now pass.

2. **But performance regression is severe** (+10% on 7 large-T workloads, even on the stride-win 4c46a94b). The full 2048-TopK index pre-scan is the culprit:
   - `tl.arange(0, 2048)` + `tl.load` loads 8 KB of int32 into registers per CTA (8 CTAs × 8 KB = 64 KB per token).
   - `tl.sum` across 2048 elements is a large reduction — many warp-level shuffles.
   - Register pressure likely spills.
   - This overhead dwarfs any HBM-coalescing benefit from contiguous idx access.

3. **Compact-block is theoretically sound but practically infeasible** in the current kernel shape. The pre-scan requires seeing all 2048 positions up front, which costs more than the coalescing saves. To pay for compact-block, we'd need:
   - A separate pre-pass kernel that computes num_valid_total per token and writes a small [T] tensor — adds a launch, adds sync.
   - Or a persistent kernel that amortizes pre-scan across multiple tokens — large rewrite.

4. **Stride-partition's strength: no pre-scan needed.** Under stride, `num_valid` = local count of valid entries within the SPLIT_SIZE=256 positions this split owns. That's a 256-element reduction, 8× cheaper than full-TopK. Even though stride has slightly worse K-row coalescing, it wins net.

## Verdict

**Reverted to exp_26.** Axis closed — compact-block partition cannot beat stride-partition without a structural kernel change (persistent, or two-launch).

## Next directions

- **Axis pivot needed.** Last two experiments (27, 29) attempted the compact-block angle; both dead. exp_28's cache modifier axis also saturated.
- **Research agent due** — 3 experiments since the exp_26 plan, 4 attempted changes, 3 reverts. Clearly in search-mode territory. Launch research for exp_30.
- Other candidates if not pivoting to research: num_stages=1 on split kernel (exp_19 tried =3 regressed; untested =1 direction), BLOCK_N_FUSED sweep on small-T kernel, or attempt Gluon rewrite per skill guidance (28 experiments in, plateau was briefly broken by exp_26 but since then only reverts).
