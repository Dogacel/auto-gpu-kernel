---
exp: 35
date: 2026-04-17
status: reverted
parent: exp_33
---

# Experiment 35 — 2026-04-17

**Description:** Change `tl.load(bt_ptrs, mask=in_bounds)` to
`mask=final_mask` in `radix_topk_kernel`. Hypothesis: masking with
`final_mask` (~2048 true lanes) instead of `in_bounds` (up to 8192 true lanes)
lets Triton/hardware skip HBM loads for positions not selected into top-K,
reducing block_table read traffic by ~4×.

## Implementation

One-line change at line 234:

```python
# Was:   global_page = tl.load(bt_ptrs, mask=in_bounds, other=0).to(tl.int64)
# Now:   global_page = tl.load(bt_ptrs, mask=final_mask, other=0).to(tl.int64)
```

`final_mask` is a strict subset of `in_bounds` (an unbounded element can never
pass the radix threshold when real scores are positive), so correctness is
preserved.

## Results

- Pass: 2/2 quick (correctness holds — final_mask ⊆ in_bounds for all bounds)
- A/B vs exp 33 (stride 8, paired same-VM):
  - B wins 10/16, mean Δ = **-0.0000 ms → tied**
  - Slow-path mixed: 6 improved (-0.5 to -1%), 2 regressed
    - Regressed: **a876010b +1.17%** (mp=91, BLOCK_N=8192 — the key workload)
    - Regressed: f457feb2 +0.93%
    - Improved: 19e7663d, 2f3b7321, 4c7705ad, 7f1cd9c2, de54c4e6, e63194e7
  - Fast-path: within ±0.3% noise
- Mode: ab-vs-exp_33
- **Reverted.** a876010b regression is the deciding factor.

## Learnings

1. **Fine-grained masks don't reliably skip HBM loads on B200.** Triton's
   `tl.load(mask=...)` semantics guarantee correct results for masked-out
   lanes, but do NOT guarantee suppressed HBM traffic for per-lane masks
   that vary within a warp. With `in_bounds` the mask is warp-uniform above
   a threshold (all lanes below max_scored are valid); with `final_mask` the
   mask varies per-lane which breaks warp-coalesced load coalescing. Result:
   HBM traffic may not decrease, and load pattern coalescing may worsen.
2. **Mixed result suggests small variance in cache behavior.** Hot L2 pages
   of block_table (each batch's ~89 pages = ~360 B) typically fit in cache.
   With different load patterns (mask=final_mask rather than in_bounds),
   cache hit timing shifts, causing the observed ±1% jitter across workloads.
3. **a876010b regression rules out as a tuning axis.** The largest slow-path
   workload is the most visible when judging an optimization — consistent
   regression there outweighs small wins elsewhere.

## Next candidates

- **Fusion attempt (one program per batch, streaming scores to registers).**
  Still the biggest ceiling (~8 µs/slow). Blocked by Triton's lack of
  register slice-assign on tiles. Needs a creative workaround or Gluon.
- **num_warps tuning on scoreless_kernel / fast_small_kernel** — haven't
  explored; they're short programs with many batches, might benefit from
  occupancy tuning.
- **Diagnose score_kernel's HBM gap** (66 → 103 GB/s). Profile suggests
  ~1.5× bandwidth headroom; figure out what's limiting (latency vs coalescing
  vs occupancy).
- **Call research agent** if next 1-2 experiments don't produce a win —
  we'd be at 5+ without improvement since exp 33.
