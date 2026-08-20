# Experiment 2 — 2026-04-16

**Description:** Dropped the in-kernel fp8 → bf16 cast; `tl.dot` now
runs directly on fp8 tensor cores. Implementation of `plan.md`. Only
the dot path changed.

## Results
- Pass: 16/16
- Kernel latency (ms): small=0.954 mean (0.389/1.354 min/max);
  large=3.405 mean (2.260/4.810); overall=min 0.389 / mean 2.179 /
  median 1.74 / max 4.810
- Reference latency (ms): n/a (profile_baseline=False)
- Max abs err: 0.00e+00  |  Max rel err: 0.00e+00 (exact match)
- Mode: stride 8

## Delta vs exp_1
- Mean: 2.408 → 2.179 ms (−9.5%)
- Small group mean: 1.057 → 0.954 (−9.7%)
- Large group mean: 3.758 → 3.405 (−9.4%)
- All 16 workloads improved (−1.9% to −12.2%); the smallest delta
  (−1.9% on `a876010b`) is near noise, but 15/16 are well above 5%.

## Learnings
- **Blackwell fp8 tensor cores work in Triton** with
  `tl.dot(fp8, fp8, out_dtype=tl.float32)` — no special casting or
  format hints required. This is the cheapest path to the tensor
  cores and was essentially free to land.
- Correctness unchanged (matched_ratio = 1.0 across all trials),
  confirming fp8 accum-to-f32 is as accurate as bf16 for this
  workload.
- Consistent ~10% latency reduction across small AND large workloads
  suggests the dot is only one of several costs; the Q/K load + post
  (ReLU, scale, weight, sum) + store pipeline accounts for the other
  90%. Future wins should target (a) memory traffic (fuse top-K
  so we don't write/read the full `scores` buffer), (b) better tile
  sizes / num_warps, (c) kernel-side top-K to eliminate the PyTorch
  per-batch loop.
