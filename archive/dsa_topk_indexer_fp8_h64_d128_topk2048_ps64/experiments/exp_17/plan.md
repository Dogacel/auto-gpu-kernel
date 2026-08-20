# Experiment 17 — Skip `torch.as_strided`, pass raw dtype-reinterpreted tensors

## Goal

Eliminate two `torch.as_strided` calls from the hot path (per-call
Python dispatch overhead) and reduce `py_setup` from 12.5 µs.

## Evidence

From `experiments/profile.md`:
| Phase | p50 across all targets |
|---|---:|
| py_setup (shape + as_strided views) | 12.5 µs flat |

The profiler noted specifically:
> py_setup (12.5 µs) — host Python dispatch for view construction;
> can't eliminate but could be trimmed by merging the two as_strided
> calls into one custom layout cache.

Each `torch.as_strided` does a PyTorch dispatch + tensor wrapper
allocation (~1.5-2 µs). Two of them = ~3-4 µs recoverable.

## Approach

**Don't create view tensors at all.** The Triton kernel only needs:
- A dtype-correct base pointer (for pointer arithmetic + load dtype)
- Explicit strides (we already pass as runtime args)
- An offset in elements (where the scales start within the raw storage)

Both views currently do *nothing* at runtime except wrap the raw
storage with new metadata. Replace with:

```python
k_fp8 = k_index_cache_fp8.view(torch.float8_e4m3fn)  # dtype reinterpret, no copy
k_scale = k_index_cache_fp8.view(torch.float32)       # dtype reinterpret, no copy
```

Then bake the scale's **byte offset** into the kernel as a compile-time
constexpr. Current scale view had `storage_offset=2048` fp32 elts; we
now add that offset inside the kernel:

```python
s_off = page_id * stride_ksp + SCALE_OFFSET + t_offs * stride_kst
```

Where `SCALE_OFFSET = page_size * head_dim // 4 = 2048` (fixed by spec).

## Additional micro-tweaks

- Hardcode known-fixed constants: `page_size=64`, `head_dim=128`,
  `head_dim_sf=132`, `topk=2048`. These are fixed by task spec
  ("dsa_topk_indexer_fp8_h64_d128_topk2048_ps64"). Skip `.shape`
  unpacking for these axes.
- Keep `batch_size` and `max_num_pages` as dynamic reads (these vary
  across workloads).

## Risks

- **Correctness**: the pointer arithmetic must match exactly. The raw
  storage is `[P, 64, 1, 132]` int8 = 8448 bytes per page. Reinterpreted
  as fp32, that's 2112 fp32 per page. The scales are at fp32 offset
  `2048..2112` within each page, which is what `SCALE_OFFSET=2048`
  encodes.
- **Triton constexpr-offset in load**: adding a constexpr to a runtime
  `s_off` tensor is standard — no issue expected.
- **Layout**: the fp8 region is still `(page_bytes=8448, head_dim=128, 1)`
  strides — same as before. Only the scale pointer base differs.

## Success criterion

- Correctness: 128/128 exact match.
- A/B vs exp 10: at least 9/16 wins (noise tolerance), mean Δ ≤ -2 µs.
- Best case: ~3-4 µs saved from skipping two as_strided dispatches.

## Followup if it wins

- Further reduce py_setup by caching `k_fp8` + `k_scale` view objects
  (they're views of a single KV cache tensor that doesn't change).
- Reduce `.stride()` call count at kernel launch (14 calls currently).
