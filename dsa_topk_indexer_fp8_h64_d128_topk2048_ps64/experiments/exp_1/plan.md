# Experiment 1 — First Triton score kernel (replace PyTorch)

## Goal

Move the score computation (`sum_h relu(Q @ K.T) * w`) from PyTorch to a
fused Triton kernel. Keep `torch.topk` for top-K selection and page
remapping — those are separate axes for later experiments.

## Why this first

Baseline PyTorch materializes `K_all` (dequantizing the **entire** cache,
including unused pages) before per-batch loops. That's wasted compute
and bandwidth on any workload where block_table touches a small subset
of pages. Moving to Triton lets us:

- Only dequantize pages actually referenced by `block_table`.
- Fuse scale-application into the dot product (post-scalar).
- Avoid materializing intermediate [H, T] or [T, D] tensors in HBM.

Even a naïve bf16-based Triton kernel should dominate the PyTorch path.

## Design

- Grid: `(batch_size, max_num_pages)`. One program per (batch, page).
- `BLOCK_H = 64`, `BLOCK_D = 128`, `BLOCK_T = page_size = 64`.
- Each program:
  - Loads `Q[b, :H, :D]` (fp8).
  - Loads `page_id = block_table[b, pid_p]` (masked to 0 if tile is
    beyond `seq_lens[b]` so we never read garbage K pages).
  - Loads `K[page_id, :T, :D]` (fp8).
  - Casts both to bf16 (safe path; fp8 tensor-core dot is exp 2).
  - `scores = Q @ K.T` via `tl.dot(q_bf, tl.trans(k_bf), out_dtype=f32)`.
  - Applies per-token scales, ReLU, per-head weights.
  - `tl.sum(axis=0)` over heads → `[T]` scores.
  - Masks out-of-bounds tokens to `-1e30` and stores to
    `scores[b, pid_p * 64 : (pid_p+1) * 64]`.
- Host: SOA fp8/scale extraction (contiguous copy, accept overhead),
  launch kernel, per-batch `torch.topk`, global-index remap.

## Expected outcome

- Correctness: should pass within `abs_err ≤ 0.02` (bf16 precision).
- Latency: very large win over PyTorch on any non-trivial workload.

## Follow-ups (future experiments)

- Skip `.contiguous()` on fp8/scale extraction (plumb strides into
  Triton instead).
- FP8 tensor-core dot (drop the bf16 cast for ~2x tensor throughput).
- Fuse top-K into the kernel (eliminate scores buffer + `torch.topk`
  launch).
- Page-level work distribution (variable launches per batch) to avoid
  wasted programs for short sequences in mixed batches.
