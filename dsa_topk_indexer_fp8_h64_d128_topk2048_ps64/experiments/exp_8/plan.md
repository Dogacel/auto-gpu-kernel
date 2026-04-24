# Experiment 8 — Triton top-K kernel (eliminate torch.topk)

## Goal

Replace `torch.topk(scores, 2048, dim=-1)` with a Triton kernel. Per
`experiments/profile.md` (a7895bf), torch.topk is **44-50% of event total
latency (45-55 µs)** — the single largest remaining phase.

## Context

After exp 7, the total phase breakdown is roughly:
- `py_setup` (view construction + scores alloc): 21 µs (19%)
- `score_kernel`: 32-48 µs (33%)
- `torch.topk`: **45-55 µs (45-50%)** — target
- `remap_kernel`: 7 µs (6%)

`stub_topk` ceiling (replace torch.topk with zero-fill): total drops
from 104 → 51 µs, i.e. ~53 µs recoverable from this phase. If a Triton
top-K can match even half of that gap, cupti-measured total should go
from 49 µs toward **~30 µs**.

## Approach

**One program per batch**, full scores tile in registers/SMEM, tile
sort via `tl.argsort` (descending), store first `effective_topk`
indices.

```python
@triton.jit
def topk_kernel(
    scores_ptr,       # float32 [B, max_scored]
    topk_idx_ptr,     # int64 [B, effective_topk]
    stride_sb, stride_st,
    stride_ib, stride_ik,
    max_scored,
    effective_topk,
    BLOCK_N: tl.constexpr,  # next pow2 ≥ max_scored
):
    pid_b = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < max_scored
    score_ptrs = scores_ptr + pid_b * stride_sb + offs * stride_st
    scores = tl.load(score_ptrs, mask=mask, other=-float('inf'))
    sorted_idx = tl.argsort(scores, dim=0, descending=True)
    # Store first effective_topk of sorted_idx
    out_offs = tl.arange(0, TOPK_BLOCK)
    out_mask = out_offs < effective_topk
    out_ptrs = topk_idx_ptr + pid_b * stride_ib + out_offs * stride_ik
    # We need the permutation values (original indices), which is sorted_idx itself
    # But we only want the first effective_topk
    # Pattern: load a subset via a mask on sorted tile
    ...
```

**Key detail:** `tl.argsort(x, descending=True)` returns the
permutation: position `k` contains the original index of the `k`-th
largest value. We just need `sorted_idx[0 : effective_topk]`.

For shape constraints (Triton tile shapes must be constexpr), we need
BLOCK_N as a compile-time constant. Since `max_num_pages` varies across
workloads (up to 89, so max_scored ≤ 89*64 = 5696), we pick the smallest
power-of-2 ≥ max_scored for each kernel launch. Options:
- **Option A:** Autotune over BLOCK_N ∈ {2048, 4096, 8192}, branch at
  launch time based on actual max_scored. A handful of compiled
  variants cover all workloads.
- **Option B:** Single fixed BLOCK_N = 8192 (covers all). Wasteful
  for small workloads but eliminates autotune complexity.

Start with **Option B**: simpler, one compiled kernel. Switch to
autotune only if we see a regression on small workloads.

## Risks

1. **`tl.argsort` support on BLOCK_N=8192** — Triton sort is typically
   bitonic/merge-based. BLOCK_N=8192 requires many warps and could
   have register-pressure or SMEM issues. If compile fails or is slow,
   fall back to tile-merge approach:
   - Load 2048-tile at a time, sort, merge with running top-2048.
   - 3 iterations for max_scored=5696.

2. **Correctness: tie-breaking.** When multiple scores are equal,
   torch.topk picks by some order (typically lowest index first on
   ascending sort; unclear for descending). Our output must match
   element-wise because validation is elementwise (abs_err/rel_err on
   `topk_indices`). **Hope:** scores from the real attention workload
   are unlikely to have exact ties (FP32 precision). If ties break
   differently, we'd see ~0 abs_err most of the time but occasional
   mismatches.

3. **Sorting overhead scales with BLOCK_N**, not max_scored. Using
   BLOCK_N=8192 for a workload where max_scored=128 (B=1, small) is
   wasteful but still fast (launch overhead dominates for tiny
   batches anyway).

## Success criterion

- A/B vs exp 7 on stride 8: B wins ≥13/16, mean Δ ≤ −15%.
- Target: mean 0.049 → 0.035 ms (~30% speedup), with larger absolute
  wins on batches where torch.topk is slowest.
- Correctness: `matched_ratio = 1.0` on all 128 workloads.

## Fallback if blocked

If `tl.argsort` doesn't compile or produces wrong results:
1. Try the tile-merge variant (load 2048, sort, merge, repeat 3x).
2. If still blocked, try packed (score, index) int64 sort via `tl.sort`.
3. If fundamentally stuck, try autotune on score_kernel (revisit
   `num_warps`, `BLOCK_T=128`). Smaller win but safer.
