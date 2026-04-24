# Experiment 20 — Workload-specialized fast path for `max_num_pages == 1`

## Goal

Cut ~40-50 µs per call on the 15 workloads that have `max_num_pages == 1`
(all batch rows fit in one 64-token page). On these workloads, current
exp 10 pipeline pays full torch.topk + remap dispatch overhead for only
~5-10 µs of real compute. A fused Triton kernel that does
score + sort + remap in one launch can skip both torch.topk (~25 µs
dispatch) and remap_kernel (~19 µs dispatch).

## Evidence

From `experiments/workload_profile.md`:
- 15/128 workloads have `max_num_pages == 1` (all batches ≤ 64 tokens).
- These include 30cecff1 (B=1), 4667f9ad (B=2), 46f236c0 (B=4),
  9410ad1e (B=6), 752c2ee5/d0c00dd5/abc9d12c (B=15/14/29 all-sl=1).

From `experiments/profile.md` on 30cecff1 (small):
| Phase | µs |
|---|---:|
| py_setup | 12.4 |
| alloc | 9.4 |
| score_kernel | 25.9 |
| torch.topk | 25.6 |
| remap_kernel | 19.2 |
| TOTAL | 93 |

On these workloads `torch.topk` is sorting ≤ 64 real elements (the
rest padded -1e30). The 25 µs is pure dispatch, not compute.
`remap_kernel` launch is also dispatch-bound (only 8 programs per B).

## Approach

Python branch on `max_num_pages == 1` (known at host side, no sync):

```python
if max_num_pages == 1:
    fast_small_kernel[(batch_size,)](
        q, k_fp8, k_scale, weights, seq_lens, block_table,
        topk_indices, ...
    )
else:
    # exp 10 path unchanged
    score_kernel(...)
    torch.topk(...)
    remap_kernel(...)
```

The fused `fast_small_kernel` (one program per batch):

1. Load seq_len, block_table[b, 0] → page_id (single entry).
2. Load Q [64, 128], K [64, 128], scale [64], w [64].
3. Compute scores [64] via the exp 10 formula (dot, relu, w-mul, sum*scale).
4. Mask positions ≥ seq_len to -∞.
5. Sort [64] via packed `uint64 = (monotone_f32 << 32) | index`
   (BLOCK_N = 64 is well within tl.sort's proven-safe range per
   LESSONS.md "tl.sort scales badly past 2048").
6. Remap: `token_idx = page_id * 64 + sorted_idx`.
7. Write output[B, 2048]:
   - First store: full [2048] = -1 (fast, single store).
   - Second store: first 64 = `where(i < actual_topk, token_idx, -1)`.

## Key design decisions

- **Branch on host-known `max_num_pages == 1`, not a GPU-synced value.**
  Per LESSONS.md ".item() sync is a structural barrier", we cannot
  branch on anything derived from GPU tensors without paying 30-60 µs.
  `max_num_pages = block_table.shape[1]` is pure Python.
- **Single program per batch**: grid is `(B,)`. No pid_p, no grid-halving
  game. For B=29 (worst case max_pg=1), 29 programs — tiny.
- **No separate remap_kernel call**: the fast kernel writes
  `topk_indices` directly, saving ~19 µs of remap dispatch.
- **No separate alloc**: `topk_indices` is pre-allocated (DPS input).
  No `torch.empty((B, max_scored))` needed.
- **Full-row -1 fill in same kernel**: one `tl.store([2048])` of -1,
  then overwrite first 64 with real values. Two stores per batch
  total, vs current path's B × max_num_pages × 1 = B write-stripe pattern.

## Risks

- **R1 (low): tl.sort on [64] uint64.** BLOCK_N=64 is far below the
  2048 wall. Should work cleanly.
- **R2 (low): constant `-1` fill of [2048] fp32 per program.**
  8 KB register tile per program. At 128 threads/warp × 4 warps = 512
  threads, each holds 4 int32 = 16 bytes. Trivial.
- **R3 (medium): adding a Python `if max_num_pages == 1` branch adds
  ~0.3 µs overhead on the 113 workloads that DON'T enter the fast
  path.** If the fast path saves 40 µs × 15 workloads = 600 µs total,
  and costs 0.3 µs × 113 = 34 µs on the rest, net win ~566 µs across
  128 workloads = **~4.4 µs mean improvement**.
- **R4 (low): correctness of the sort-and-remap.** Exp 8 showed
  packed `(monotone_bits << 32) | idx` works for descending fp32
  sort. Just reuse.

## Success criterion

- Correctness: 128/128 exact match.
- A/B vs exp 10: mean Δ ≤ -1 µs, at least 15/16 workloads within noise
  (max_num_pages=1 workloads win big; others ≈ tied from branch cost).
- Quick (both workloads have max_num_pages > 1 AFAIK — need to check).

## Expected magnitude

Per-workload on fast-path hits (15 workloads): **40-50 µs savings**.
Across all 128 workloads (A/B only samples 16): likely 0-1 fast-path
hits in the stride-8 sample, so mean Δ is dominated by the cost of
the Python branch on non-fast-path workloads (~0.3 µs × 15 = 5 µs
net positive on the sample, hopefully).

Full-run (128 workloads): if 15 workloads drop 40 µs each and 113
workloads see no change, mean improvement = 15*40/128 = **4.7 µs** or
-10% from 47 µs baseline.

## Followup if it wins

- Extend fast path to `max_num_pages == 2` (covers more workloads).
- Consider fast path for `active_programs <= 8` generally (requires
  seq_lens info which is GPU-only, but a coarse check via `max_num_pages <= 2`
  covers most of the "trivial work" cases).
