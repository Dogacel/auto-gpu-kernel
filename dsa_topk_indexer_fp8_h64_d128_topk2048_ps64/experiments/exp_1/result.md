# Experiment 1 — 2026-04-16

**Description:** First Triton kernel. Replaced PyTorch score compute
with a fused Triton kernel: per-(batch, page) program loads `Q[b]`,
`K[page_id]`, per-token scales, and per-head weights; computes
`relu(Q @ K.T) * scale[t] * w[h]`, sums over heads, stores
`[B, max_num_pages * page_size]` scores buffer. FP8 inputs cast to
bf16 for `tl.dot` (fp8 tensor-core path deferred to exp 2). Top-K
selection + global index remapping remains in PyTorch (per-batch).
See `plan.md`.

## Results
- Pass: 16/16
- Kernel latency (ms): small=1.057 mean / 1.01 median (first 8 workloads);
  large=3.758 mean / 3.68 median (last 8 workloads);
  overall=min 0.408 / mean 2.408 / median 1.98 / max 5.002
- Reference latency (ms): n/a (profile_baseline=False)
- Max abs err: 0.00e+00  |  Max rel err: 0.00e+00 (exact match — top-K
  indices matched 100% across all trials)
- Mode: stride 8

## Learnings
- Starting point: the PyTorch baseline is fully replaced. Absolute
  latencies now land at 0.4–5.0 ms across a 16-workload spread.
- FP8 → bf16 cast inside the kernel retained exact correctness
  (`max_abs = 0.00e+00`, `matched_ratio = 1.0`). bf16 has ample
  precision for fp8 dot inputs; no need for f32 dot or tf32x3 path
  for now.
- `tl.where(tile_active, page_id, 0)` is the key safety trick — it
  keeps K loads in bounds for block_table slots past `seq_lens[b]`
  without requiring a `return` early-exit.
- Host-side `.contiguous()` on fp8/scale extraction is still present
  and costs ~8 μs per call at 1000 pages. Next experiment should
  plumb the SOA strides through to Triton instead of pre-copying.
- Per-batch Python `torch.topk` loop is cheap for small batch counts
  but will dominate once the kernel is fully optimized. Fusing top-K
  is a later axis.
- First real win axis candidates: (a) FP8 tensor-core dot, (b) skip
  `.contiguous()`, (c) kernel-side top-K, (d) persistent / variable
  grid sizing for mixed-seq-len batches.
