# Experiment 19 — Two-page per program via TWO independent dots (serial store)

## Goal

Preserve exp 18's grid-halving benefit (wins on smallest -16%, largest
-6%) while eliminating the medium-workload regressions (+5-7% on 6
workloads) caused by `tl.trans` + `tl.reshape` layout conversion of
the concatenated K tile.

## Hypothesis

The medium regression in exp 18 was driven by the K-tile layout
materialization: `tl.join(k0, k1)` → `tl.trans(2, 0, 1)` → `tl.reshape([128, 128])`
forces a data rearrangement on 16 KB of fp8 data per program. This
overhead dominates on medium workloads (B≈16-25, max_num_pages≈15-35)
where launch-overhead savings are modest.

## Approach

Keep two pages per program but do TWO independent [64, 64×128] MMAs
(same as exp 10 structure, repeated twice) with STORE-BETWEEN pattern
that lets registers free up between iterations:

```python
# First page
page_id_0 = load(block_table[b, pid_p * 2])
k0 = load(K[page_id_0])
s0 = load(scales[page_id_0])
scores_0 = dot(q, k0.T)  # [64, 64]
reduced_0 = sum(relu(scores_0) * w) * s0
store(scores[b, token_start:token_start+64], reduced_0)

# Second page (clamp for odd max_num_pages)
page_id_1 = load(block_table[b, min(pid_p * 2 + 1, max_num_pages - 1)])
k1 = load(K[page_id_1])
s1 = load(scales[page_id_1])
scores_1 = dot(q, k1.T)
reduced_1 = sum(relu(scores_1) * w) * s1
store(scores[b, token_start+64:token_start+128], reduced_1)
```

Q and `w` are loaded once per program (amortized). Each matmul's register
state gets freed after its store, so peak register pressure is the same
as exp 10 (single [64, 64] accumulator).

## Key difference from exp 3 (failed PPP>1 loop)

Exp 3 used `tl.static_range(PPP)` for a generic N-page loop. That failed
with +85% regression on large workloads. This implementation:
- Full unrolls the 2-page case (no tl.range/tl.static_range) — each
  iteration is explicit code.
- Serializes stores to free register state between iterations.
- Shares Q/w loads (1 load per program instead of 2).

Exp 3's LESSON: "Don't revisit unless we change the tile shape." This
experiment does change the tile shape (BLOCK_T logically = 128, not
a loop of 64×N); the cost per iteration is the same as exp 10's
original body.

## Risks

- **R1**: Compiler may CSE the two dots into the same-shape MMA pattern
  that LESSONS warns about. Mitigation: explicit `tl.store` between
  iterations ensures dataflow barrier.
- **R2**: Losing across-program pipelining is unavoidable with halved
  grid. But exp 18's data shows this is NET positive on extremes;
  the regression came from tl.trans+tl.reshape, not grid halving.
- **R3**: Two separate [64, 128] K loads may not amortize HBM bandwidth
  as well as a single [128, 128] load. But HBM latency usually
  overlaps issue of the second load.

## Success criterion

- Correctness: 128/128 exact match.
- A/B vs exp 10: mean Δ ≤ -1 µs, at least 10/16 wins.
- **No workload regresses >5%** (exp 18 violated this on medium).

## Expected magnitude

Best case: preserves exp 18's wins (smallest -16%, largest -6%) while
neutralizing medium regressions → mean ~-3 µs (-6%). Realistic case:
mean -1 to -2 µs (-2 to -4%).
