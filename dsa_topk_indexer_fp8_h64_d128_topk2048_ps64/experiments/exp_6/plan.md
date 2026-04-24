# Experiment 6 — Skip `.contiguous()` on K cache (strided fp8/scale views)

## Goal

Eliminate the ~210 µs HBM-bound `.contiguous()` copy of the entire FP8
K cache + the ~13 µs scale copy. Per the profile, these are 62% of total
latency (225 µs of 360 µs). Pass strided views to the kernel instead.

## Why this differs from exp 3's skip-`.contiguous()` ablation

Exp 3 tested with the per-batch Python loop still in place. That loop
added ~2 ms of overhead on large workloads, which dominated and masked
the win. The report concluded "large regressed" but it was measuring
a totally different post-kernel path.

Post-exp-4, the Python loop is gone. Score kernel is now only 11–27 µs
even for the worst workload. A 2× kernel slowdown from strided loads
costs ≤30 µs but saves ~220 µs of setup. Net win ~190 µs.

## Change

Host:
```python
# Before (exp 4):
kv_u8 = k_index_cache_fp8.view(torch.uint8).reshape(num_pages, page_size * head_dim_sf)
fp8_view = kv_u8[:, :page_size * head_dim].contiguous().view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
scale_view = kv_u8[:, page_size * head_dim:].contiguous().view(num_pages, page_size, 4).view(torch.float32).squeeze(-1)

# After (exp 6): direct strided views, zero copy
kv_fp8 = k_index_cache_fp8.view(torch.float8_e4m3fn)          # [P, 64, 1, 132]
fp8_view = kv_fp8[:, :, 0, :head_dim]                         # [P, 64, 128] strided (stride_t=132)
kv_f32 = k_index_cache_fp8.view(torch.float32)                # [P, 64, 1, 33] — 132B = 33 f32
scale_view = kv_f32[:, :, 0, head_dim // 4]                   # [P, 64] float32, strided (stride_t=33)
```

Kernel: unchanged. The existing kernel already receives strides as
arguments — we're just passing different strides.

## Risk / expected kernel slowdown

- **Strided fp8 load:** stride_t = 132 bytes (non-power-of-2). 64
  tokens × 128 bytes = 8 KB data spread over 8.25 KB address range.
  Inner d-dim is contiguous, so warp-level coalescing on d is
  preserved. Token-level access crosses 128-byte lines inefficiently.
- **Strided scale load:** 64 floats with stride 33 × 4 = 132 bytes.
  Each float is 4 bytes in a 132-byte line, so 128 of those bytes are
  actually the fp8 data we already loaded. L2 should absorb this.
- **Max kernel slowdown estimate:** ~2× worst case = 22-54 µs.

## Success criterion

- /benchmark quick passes correctness (exact match).
- A/B vs exp 4 on stride 8: B wins ≥12/16 with mean Δ ≤ −20%. Target
  is ~0.15 ms mean (down from 0.31 ms).
- If the kernel regresses significantly but setup savings still win:
  keep. If the combined effect is neutral or worse: need a different
  approach (in-kernel SOA unpack with packed int32 loads).
