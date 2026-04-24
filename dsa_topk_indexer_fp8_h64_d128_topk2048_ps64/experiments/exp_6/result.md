# Experiment 6 — 2026-04-16

**Description:** Eliminated the two `.contiguous()` memcpy copies of
the FP8 K cache (fp8 data + scales) that the profiler identified as
~62% of total latency. Replaced them with zero-copy `torch.as_strided`
views that point directly at the SOA layout in `k_index_cache_fp8`.

## Results
- Pass: 128/128 (exact match)
- Kernel latency (ms): min 0.188 / mean 0.200 / median 0.200 / max 0.212
- Max abs err: 0.00e+00  |  Max rel err: 0.00e+00
- Mode: full (128 workloads)

## A/B vs exp 4 (same-VM, stride 8)
- B wins 16/16
- mean Δ = −0.156 ms (−44% avg per workload)
- per-workload range: −37.9% to −48.5%

## Delta vs exp 4 (full)
- exp 4 mean = 0.310 ms → exp 6 mean = 0.200 ms (−35.5%)
- max: 0.342 → 0.212 ms (−38%)
- min: 0.284 → 0.188 ms (−34%)

## What made exp 3's skip-`.contiguous()` fail and this one succeed

Exp 3 tried sliced views like `kv_u8[:, :8192].view(...)` which rely
on `.view()` checking contiguity — that failed silently (wrong stride
interpretation) and interleaved scales with fp8 data, producing either
garbage or compile errors. Exp 6 uses `torch.as_strided` with explicit
shape/stride/offset, carving out the SOA regions directly:

- fp8: `stride=(page_size*head_dim_sf=8448, head_dim=128, 1)`. The
  intra-page stride is **still 128** (the fp8 block per token is
  contiguous in memory). Only the cross-page stride is 8448 (vs 8192
  for the copied version) — irrelevant because each program reads
  exactly one page.
- scale: storage_offset=2048 fp32 elements (= byte 8192) puts the
  view at the scale region; stride=(2112, 1) matches the per-page
  scale block.

Because intra-page strides match the contiguous case, the score
kernel's memory access pattern is **unchanged**. We only pay different
base pointers per page.

## Learnings
- **The previous lesson in LESSONS.md was wrong.** "stride 132 hurts
  vs 8192" applied to a broken AOS interpretation of the layout.
  Correctly-strided views cost nothing — the SOA fp8 data is already
  contiguous per page, `as_strided` just skips the cross-page copy.
- **Profile-guided wins are high-yield.** Profiler named the exact
  phase (setup = 62% of 360 µs) and predicted the ceiling (~150 µs
  if setup eliminated). Landed within 30% of that target.
- The remaining 200 µs is now dominated (per the profile) by:
  topk ~46 µs, remap ~38 µs, mask_write ~36 µs, score_kernel ~15 µs.
  Next axes: fuse remap/mask_write into kernel, or reduce torch
  launch count in the post-processing path.

## Next candidates
- **Merge topk output remap + mask into one kernel call** (fewer torch
  launches → save ~20–40 µs of overhead).
- **Writing sorted indices directly in a Triton kernel** (eliminate the
  scores buffer entirely; replace with kernel-side partial sort + merge).
- **Explicit num_warps tuning** if we structurally expose kernel cost.
