---
exp: 9
date: 2026-04-17
status: kept
parent: exp_7
---

# Result — Early return from inactive score_kernel programs

## Change
Added a block-uniform early-return at the top of `score_kernel` when
`token_start >= seq_len`. Writes the `-1e30` sentinel to the scores
row and returns before loading Q/K, doing the matmul, or computing
scales/weights.

```python
seq_len = tl.load(seq_lens_ptr + pid_b)
token_start = pid_p * BLOCK_T
if token_start >= seq_len:
    t_offs_sk = tl.arange(0, BLOCK_T)
    score_off_sk = pid_b * stride_sb + (token_start + t_offs_sk) * stride_st
    tl.store(scores_ptr + score_off_sk, tl.full([BLOCK_T], -1e30, tl.float32))
    return
```

Also dropped the `tl.where(tile_active, page_id_raw, 0)` guard from
exp 7 since it's dead code after early-return.

## Measurement

- `/benchmark full`: mean **0.0478 ms**, min 0.022, max 0.076, 128/128 pass
  - small n=25: mean 0.028
  - medium n=83: mean 0.048
  - large n=20: mean 0.071
- A/B vs exp 7: B wins 14/16, mean Δ = −0.0015 ms (−3%). Biggest
  wins on a876010b (−6.78%) and 2f3b7321 (−6.31%) — both have low
  `sum(seq_lens) / (B * max_num_pages * page_size)` utilization.

## Vs exp 7 (0.049 ms mean, 0.022/0.082 min/max)
- Mean: −2.4%.
- Max: 0.082 → 0.076.
- Pass: still 128/128, exact match preserved.

## Why this works
`max_num_pages` is dimensioned for the largest batch across
workloads — on a876010b (B=29, max_num_pages=89, sum_sl=8812),
only ~138 / 2581 programs have any in-bounds token. The other
~94% do a wasted 64×128 FP8 matmul that gets masked to `-1e30`.

## Why the observed win is smaller than the back-of-envelope
I estimated 30+ µs savings on a876010b; actual was ~3–6 µs (from
A/B Δ). FP8 tensor core matmul is already cheap (64×128 @
128×64 FP8 in one or two ticks on B200), so the skipped work
isn't that costly in absolute terms. What this really saved is
HBM pressure (no Q/K/weights/scales loads on 94% of tiles) and
some L2 traffic, which shows up as a modest speedup.

## Risks that didn't bite
- `return` at block level compiled fine on the Modal Triton build.
- Branch is block-uniform (`seq_len` same for all threads), so no
  warp divergence.
- Correctness path unchanged for partial-tile case — still uses
  `tl.where(abs_t < seq_len, ...)` for the in-bounds vs out-of-bounds
  mix.

## Takeaway
Block-uniform early-returns on indices derived from batch-level
metadata (here `seq_len`) are cheap wins when the grid is
dimensioned conservatively for the worst-case batch. Look for
other places where the grid is sized for the max across something
but most programs short-circuit.
