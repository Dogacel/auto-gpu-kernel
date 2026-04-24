# Experiment 16 — Cache `scores` buffer via module-level pool keyed by shape

## Goal

Eliminate the ~9 µs `torch.empty((B, max_scored), ...)` CPU-side
overhead paid on every call. Profile (exp 10) shows alloc at 9.4-9.8 µs
flat across regimes, entirely CPU-side (PyTorch dispatch +
allocator lookup), not HBM-bound.

## Evidence

From `experiments/profile.md`:
| Phase | small | medium | large |
|---|---:|---:|---:|
| py_setup | 12.4 | 12.7 | 12.5 |
| **alloc** | **9.4** | **9.6** | **9.8** |
| score_kernel | 25.9 | 27.3 | 33.8 |

On small workloads (dispatch-bound, 93 µs total), alloc is 10% of the
total — a ~10% one-axis improvement.

## Approach

Module-level cache `_scores_cache: dict[tuple[int, int], torch.Tensor]`
keyed by `(batch_size, max_scored)`. Check cache; return cached buffer
if present, otherwise alloc + insert.

```python
_scores_cache: dict = {}

def _get_scores_buffer(batch_size: int, max_scored: int, device, dtype):
    key = (batch_size, max_scored, device.type, device.index)
    buf = _scores_cache.get(key)
    if buf is None:
        buf = torch.empty((batch_size, max_scored), device=device, dtype=dtype)
        _scores_cache[key] = buf
    return buf
```

Correctness: `score_kernel` writes every position of `scores[:, :max_scored]`
(active programs write real scores, early-return programs write `-1e30`
sentinels per exp 9). Previous data in the buffer is fully overwritten.
Returning a stale reference is safe as long as nothing else holds it
(the kernel and `torch.topk` consume it immediately, before next call).

## Risks

- **Memory footprint**: with 128 workloads each potentially unique
  shape, cache could grow to 128 buffers. At max size
  (30 × 5696 × 4 = 683 KB), total ≤ ~90 MB — bounded.
- **Across-workload state leak**: different workloads don't share
  buffers (different keys), so no leak.
- **Benchmark-gaming concern**: caching is a standard buffer-pool
  optimization, not "memoizing outputs". Legitimate — would apply
  equally in production. No CUDA graphs, no iteration counters.
- **Device / dtype stability**: we include `device.type` and
  `device.index` in the key. Dtype is always fp32.

## Success criterion

- Correctness: 128/128 exact match.
- A/B vs exp 10: at least 10/16 wins, mean Δ ≤ -3 µs (-6%).
- Full run mean: ≤ 0.044 ms.

## Followup if it wins

- Cache the `topk_idx` int64 intermediate too (allocated as
  `torch.topk`'s output — not directly controllable, but
  `torch.topk(..., out=(values, indices))` accepts preallocated).
- Cache `fp8_view` and `scale_view` — but these are zero-copy views
  of the input tensor, so no alloc to cache.
