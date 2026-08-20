---
exp: 33
date: 2026-04-17
status: accepted
parent: exp_29
---

# Experiment 33 — 2026-04-17

**Description:** Merge the two `tl.cumsum` calls in `radix_topk_kernel` into
one via packed-uint32 prefix encoding. hi16 bits of each lane carry
`cumsum(strict_mask)`, lo16 bits carry `cumsum(tie_mask)`. Both halves fit
(BLOCK_N ≤ 8192 ≤ 0xFFFF). Saves one BLOCK_N-wide prefix reduction and
eliminates the data dependency between `tie_prefix → final_mask → write_prefix`.

## Implementation

In `radix_topk_kernel`'s post-loop scatter:

```python
# was:
#   strict_count = tl.sum(strict_mask)
#   tie_prefix   = tl.cumsum(tie_mask)           # cumsum 1
#   final_mask   = strict | (tie & tie_prefix <= remaining)
#   write_prefix = tl.cumsum(final_mask)         # cumsum 2
#   write_pos    = write_prefix - 1

packed = (strict_mask.to(uint32) << 16) | tie_mask.to(uint32)
packed_prefix = tl.cumsum(packed)                # ONE cumsum
strict_prefix = (packed_prefix >> 16).to(int32)
tie_prefix    = (packed_prefix & 0xFFFF).to(int32)

strict_count = tl.sum(strict_mask.to(int32))
remaining    = topk - strict_count
valid_tie    = tie_mask & (tie_prefix <= remaining)
final_mask   = strict_mask | valid_tie
# Strict → [0, strict_count); valid ties → [strict_count, topk).
write_pos    = tl.where(strict_mask, strict_prefix - 1,
                        strict_count + tie_prefix - 1)
```

Correctness: strict positions get scatter index `strict_prefix[i] - 1`
(0-indexed within the strict prefix, spanning `[0, strict_count)`); valid ties
get `strict_count + tie_prefix[i] - 1` (in `[strict_count, strict_count +
remaining) = [strict_count, topk)`). Total written positions = topk by radix
invariant.

## Results

- Pass: 2/2 quick (correctness) + 16/16 A/B (all exact, matched_ratio=1.0000)
- A/B vs exp 29 (stride 8, paired same-VM):
  - **B wins 10/16, mean Δ = -0.0001 ms → B faster**
  - **All 8 slow-path workloads improved -1.0 to -1.9%**:
    - 7f1cd9c2 -1.93% | 2f3b7321 -1.39% | e63194e7 -1.19% | 19e7663d -1.12%
    - 4c7705ad -1.10% | de54c4e6 -1.10% | a876010b -1.01% | f457feb2 -1.00%
  - Fast-path (mp=1, scoreless ≤32, radix scoreless branch): within ±0.67% noise
- Mode: quick + ab-vs-exp_29
- **New best.**

## Learnings

1. **Two serial cumsums over BLOCK_N had a measurable cost.** The 1-2% win on
   every slow-path workload indicates one full cumsum over BLOCK_N (up to 8192)
   is ~0.3-0.5 µs of kernel time. The algorithmic dependency chain
   `tie_prefix → final_mask → write_prefix` serialized both scans, so the cost
   was not amortizable via ILP.
2. **Packed-prefix trick generalizes.** When two cumsums over the same tile
   have values bounded by 2^k each, pack them into a single integer wider than
   `2k` bits and compute one cumsum. Triton's `tl.cumsum` on uint32 handles
   lane-parallel integer addition; hi/lo halves stay independent as long as
   no half overflows. Applicable to any "count both A and B up to position i"
   pattern where the counts are small.
3. **Profile's "< 1 µs secondary lever" was optimistic.** The cumsum merge
   lands at ~0.3-0.4 µs full-run mean save (extrapolating from A/B), above
   the noise floor and worth keeping even though small.

## Next candidates

- **Cross-radix bit-loop optimization**: the 32× `tl.sum` in the bit loop is
  ~6-10 µs of radix. A 4-bit radix (8 iterations) with 16-bucket histograms
  would change the reduction axis but needs Triton histogram support.
- **Fusion remains the biggest ceiling** (~8 µs per slow workload). Profile's
  register-resident streaming scores design is still the most promising
  structural lever, but needs an in-Triton representation of a [BLOCK_N]-wide
  register array with tile-offset updates (currently blocked by Triton's
  lack of slice assignment on tiles).
- `num_warps` sweep on radix's scoreless branch (currently single-program
  per batch, no reduction) — likely no-op but untested.
- Retry `num_warps=16` on radix now that one less cumsum is in flight (lower
  register pressure may permit higher occupancy).
