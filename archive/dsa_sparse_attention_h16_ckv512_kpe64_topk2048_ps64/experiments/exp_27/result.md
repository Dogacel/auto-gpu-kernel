# Experiment 27 — 2026-04-17

**Description:** Attempted compact-block partition by valid-count. Each CTA scans the full TopK=2048, computes `num_valid_total`, and owns the contiguous valid-position range `[s*nv/NUM_SPLITS, (s+1)*nv/NUM_SPLITS)`. Goal: preserve exp_26's load balance while restoring contiguous K-cache row access (expected to recover the +0.5-1.8% T=8 HBM-coalescing regressions).

## Results
- Pass: **1/2 on quick** (0c23b10c PASSED, 2207f0fd FAILED with abs_err=2.82)
- Mode: quick only — did not proceed to A/B
- **Reverted immediately to exp_26 state**

## Design (reverted)

```python
# BEFORE (exp_26 strided):
offs_split = s + tl.arange(0, SPLIT_SIZE) * NUM_SPLITS
idx_scan = tl.load(Indices_ptr + t*stride + offs_split)
num_valid = tl.sum((idx_scan >= 0).to(tl.int32), axis=0)
max_bn = ((num_valid + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
for bn in range(0, max_bn, BLOCK_N):
    idx_ptrs = ... + (s + (bn + offs_n) * NUM_SPLITS)

# ATTEMPTED (compact-block by valid count):
offs_topk = tl.arange(0, TOPK)  # full 2048
idx_scan = tl.load(Indices_ptr + t*stride + offs_topk)
num_valid_total = tl.sum((idx_scan >= 0).to(tl.int32), axis=0)
start_s = (s * num_valid_total) // NUM_SPLITS
end_s = ((s + 1) * num_valid_total) // NUM_SPLITS
range_s = end_s - start_s
max_bn = ((range_s + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
for bn in range(0, max_bn, BLOCK_N):
    idx_ptrs = ... + (start_s + bn + offs_n)
```

## Discoveries

1. **Compact-block by-valid-count produces wrong output on T=8 workloads** (abs_err=2.82, 100× above tolerance). Root cause not fully diagnosed before revert; hypotheses:
   - Partition arithmetic `(s * n) // NUM_SPLITS` may be producing unexpected values when `n` is a Triton scalar (from reduction); possible scalar-vs-tensor type confusion.
   - Or the full 2048-element scan produces a different `num_valid_total` than the per-split scans in exp_26 (unlikely — both count the same `idx >= 0` mask).
   - Possible Triton limitation on large `tl.arange(0, TOPK=2048)` with subsequent reduction to a scalar used for pointer arithmetic.
   - T=1 (0c23b10c) passed — suggests the bug is related to high `num_valid_total` values (T=8 workloads have per-token valid in the hundreds-thousands).

2. **The theoretical gain still stands.** Under prefix-contig-valid input (workload profile p50 contig=1.0), compact-block partition's K-cache-row access is strictly better than strided — rows are adjacent in HBM, preserving burst coalescing. Stride-partition fetches 128 K rows at stride-8 in KV-cache per iter, each row a separate HBM burst group. If a working implementation lands, should recover exp_26's large-T regressions without sacrificing the 4c46a94b win.

3. **Debugging path forward:** inspect PTX of both strided and compact versions on a minimal reproducer; or use `tl.device_print` (if available in Triton 3.6) for a single thread to log `num_valid_total`, `start_s`, `end_s`. Deferred — rate-limiting to keep experiment cadence.

## Verdict

**Reverted.** Correctness blocker. Keep exp_26 as the current best. The axis is promising but needs debugging.

## Next directions

- Defer compact-block; pivot to a different axis for exp_28.
- Alternative for HBM coalescing recovery: per-iter prefetch ordering, or `.cg` on strided K loads (may or may not interact with L2 per LESSON 40).
- Or: workload-specialized kernel choice — host dispatch based on some cheap-to-compute proxy of max-per-token-valid.
