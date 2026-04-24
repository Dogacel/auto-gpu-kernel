# Experiment 22 — mp=2 fast path via TWO independent `[64, 64]` dots (combine fp32)

## Goal

Retry the mp=2 fast path after exp 21's regression (+180% on target
workloads). Replace the big-MMA concat pattern (`tl.join + tl.trans +
tl.reshape` on 16 KB fp8 data) with two independent `[64, 128] @
[128, 64]` MMAs that combine their fp32 `[64]` reduction outputs
via a tiny 256 B join.

## Evidence from exp 21

5 mp=2 workloads regressed from ~25 µs to ~70 µs with the big-MMA
approach. Root cause (per `LESSONS.md`): the `tl.join + tl.trans +
tl.reshape` rearrangement on two `[64, 128]` fp8 tiles (16 KB)
requires SHMEM shuffles that aren't amortized on 1-program-per-batch
grids.

## Approach

One program per batch. Grid `(B,)`. In the kernel:

```python
# Load shared state
q_fp8 = load(Q [64, 128])
w = load(weights [64])
page_0 = load(block_table[b, 0])
page_1 = load(block_table[b, 1])
seq_len = load(seq_lens[b])

# Page 0: independent dot
k0 = load(K[page_0])        # [64, 128] fp8
s0 = load(scale[page_0])    # [64] fp32
scores_0 = tl.dot(q_fp8, tl.trans(k0), out_dtype=fp32)  # [64, 64]
final_0 = tl.sum(tl.maximum(scores_0, 0) * w[:, None], axis=0) * s0  # [64]

# Page 1: independent dot
k1 = load(K[page_1])        # [64, 128] fp8
s1 = load(scale[page_1])    # [64] fp32
scores_1 = tl.dot(q_fp8, tl.trans(k1), out_dtype=fp32)  # [64, 64]
final_1 = tl.sum(tl.maximum(scores_1, 0) * w[:, None], axis=0) * s1  # [64]

# Combine fp32 vectors (256 B, trivial)
final_joined = tl.join(final_0, final_1)  # [64, 2]
final_perm   = tl.trans(final_joined, 1, 0)  # [2, 64]
final        = tl.reshape(final_perm, [128])  # [128]

# Mask, sort, remap as before
...
```

## Key difference from exp 21

- **No fp8 layout conversion.** Two `[64, 64]` scores live naturally in
  register files — no SHMEM shuffle needed to stack them into a
  [128, 128] tile.
- **Two smaller MMAs are same total FLOPs** as one big-MMA (both are
  2 × 64 × 64 × 128 = 1M fp8 MMAs), so compute time is equivalent.
- **fp32 combine** happens on `[64]` vectors (256 B), not 16 KB fp8
  tiles. The join/trans/reshape overhead scales with tensor size.

## Key difference from exp 19

Exp 19 also did two dots but in the `score_kernel` context with
grid `(B, max_num_pages / 2)`. That halved the grid, losing
cross-program pipelining benefit (`LESSONS.md`: "PAGES_PER_PROGRAM>1
regressed big"). Exp 22 does two dots in a single-program-per-batch
grid where cross-program pipelining doesn't apply — each program
has its own work, and Blackwell's schedulers handle within-program
dual-MMA pipelining via warp-specialization.

## Risks

- **R1 (low): compiler may still fuse the two dots into a suboptimal
  structure.** Mitigation: the `final_0 → sum → * s0` reduction between
  them acts as a dataflow barrier (separate store-ish pattern exp 19
  relied on, without the grid-level cost).
- **R2 (low): `tl.sort` on 128 uint64.** Same as exp 21 — at N=128,
  well below the 2048 wall. Observed in exp 20 that N=64 sort is
  ~0.5 µs; N=128 should be ~1 µs.
- **R3 (medium): kernel may still be worse than the default path.**
  The default path on mp=2 takes ~25 µs (2-page score_kernel + small
  torch.topk + remap). If exp 22 kernel can't beat ~20 µs, no win.

## Expected magnitude

Target: fast-path kernel at ~12-15 µs (mp=1 is 8 µs; +1 dot, +1 scale
load, +1 page load, +1 join-fp32 → +4-7 µs). Savings vs default
path: ~10-13 µs per workload × 5 workloads = 50-65 µs ÷ 128 = **~0.4-0.5 µs
mean improvement**.

Small incremental but in the right direction.

## Success criterion

- Correctness: 128/128 exact match.
- mp=2 workloads drop from ~25 µs to <20 µs each (ideally ~15 µs).
- Overall mean Δ < -0.3 µs vs exp 20.
- If either mp=2 workload regresses, revert.

## Followup if it wins

- Extend approach to `max_num_pages == 3` (three dots, 4 more workloads).
- Extend to `max_num_pages == 4` (four dots, 2 more workloads).
- At some N the N-dot approach loses to the default path; find the
  crossover threshold.
