# Experiment 30 — 2026-04-17

**Description:** Research-agent plan (see `plan.md`): cache `partial_m`, `partial_l`, `partial_acc` scratch buffers across calls in `_partial_buffer_cache` keyed by `(device, num_tokens)`, extending the `_counter_cache` pattern (LESSON-22). Buffers are write-before-read so no zero-init is required. Targets host-side allocation overhead — the last untouched axis per the plan's diagnosis.

## Results
- Pass: 2/2 quick (0c23b10c, 2207f0fd)
- Mode: quick + A/B vs exp_26 (stride-2)
- **Reverted** — mild regression

**A/B run vs exp_26 (stride-2):**
| UUID | A (exp_26) | B (exp_30) | Δ | % |
|---|---|---|---|---|
| 02d6ae9c | 0.0159 | 0.0158 | −0.0001 | −0.44% |
| 05f6de65 | 0.0186 | 0.0187 | +0.0001 | +0.40% |
| 0c23b10c | 0.0052 | 0.0051 | −0.0000 | −0.44% |
| 2207f0fd | 0.0158 | 0.0159 | +0.0001 | +0.49% |
| 232ed014 | 0.0154 | 0.0155 | +0.0001 | +0.41% |
| 4c46a94b | 0.0114 | 0.0116 | +0.0002 | +1.94% |
| 5096e459 | 0.0160 | 0.0161 | +0.0001 | +0.54% |
| 564007ac | 0.0161 | 0.0162 | +0.0001 | +0.46% |
| 78b2e11c | 0.0157 | 0.0158 | +0.0001 | +0.57% |
| b7668cfd | 0.0054 | 0.0054 | +0.0000 | +0.06% |
| e6b849f2 | 0.0079 | 0.0079 | +0.0000 | +0.20% |
| f77df5ce | 0.0053 | 0.0053 | −0.0000 | −0.24% |

Paired: B wins 3/12, mean Δ = +0.0000 ms (A faster). 4c46a94b regressed +1.94% (the stride-win workload).

## Design (reverted)

```python
_partial_buffer_cache: dict = {}

def _get_partial_buffers(num_tokens, device, NUM_SPLITS, H, D_ckv):
    key = (device, num_tokens)
    cached = _partial_buffer_cache.get(key)
    if cached is None:
        partial_m = torch.empty((num_tokens, NUM_SPLITS, H), dtype=torch.float32, device=device)
        partial_l = torch.empty((num_tokens, NUM_SPLITS, H), dtype=torch.float32, device=device)
        partial_acc = torch.empty((num_tokens, NUM_SPLITS, H, D_ckv), dtype=torch.float32, device=device)
        cached = (partial_m, partial_l, partial_acc)
        _partial_buffer_cache[key] = cached
    return cached

# Replace lines 348-350 with:
partial_m, partial_l, partial_acc = _get_partial_buffers(num_tokens, device, NUM_SPLITS, H, D_ckv)
```

## Discoveries

1. **Host-side `torch.empty` overhead is invisible to CUPTI.** The plan's hypothesis ("CUPTI captures cudaMalloc*/cudaFree* runtime calls") is falsified. PyTorch's caching allocator either (a) never round-trips to the CUDA runtime for these small sizes after warmup, or (b) the runtime calls it does make happen outside CUPTI's measurement window. Either way, the three `torch.empty` calls cost effectively zero measurable microseconds.

2. **The `_counter_cache` pattern does NOT generalize.** LESSON-22 measured ~5 µs regression from a single `torch.zeros` per call; that cost was specifically the `cudaMemset` zero-init, not the allocation. `torch.empty` skips the memset, so caching its output saves nothing measurable.

3. **Mild consistent regression (+0.4-0.6% on most workloads) from the helper function overhead.** The `_get_partial_buffers` call adds a dict lookup + tuple unpacking vs three straight-line `torch.empty` calls. This tiny Python overhead is visible on the stride-win workload `4c46a94b` (+1.94%) where the kernel is already extremely fast (0.012 ms) so relative Python overhead matters more.

4. **Host-overhead axis closed.** The combined hit rate (direct allocation + counter-only caching) is the local optimum. Adding more caching infrastructure doesn't help.

## Verdict

**Reverted to exp_26.** The "untouched axis" from the research plan turned out to not be an axis at all — allocator caching already handles it. Clean negative result; rules out host-side Python overhead entirely.

## Next directions

- **All measurable host/compiler/cache axes are now exhausted.** The remaining ~5 µs of recoverable headroom in `profile.md` sits behind structural walls (PDL invisible, cluster sync incompatible, Gluon software-FMA too slow for the full MMA path).
- **Remaining candidate axes**:
  1. **Partial Gluon rewrite** using `bw.tcgen05_mma` + `bw.mbarrier` on the Blackwell MMA path — high-effort, high-risk, 10+ iterations runway.
  2. **Re-examining the combine phase** — exp_15 atomic-barrier was the last combine-phase win. NUM_SPLITS=8 + BLOCK_D=64 is assumed optimal but worth a re-sweep under stride-partition (the sweep was done under block-partition in exp_3-5).
  3. **Fused-kernel (T≤2) structural changes** — BLOCK_N_FUSED=64 was tuned in exp_13; other small-T path knobs untouched.
  4. **Workload-specialized dispatch** — precompute per-token valid counts on host (1 tiny kernel) then dispatch to stride/block variants.
- **Next iteration plan**: try NUM_SPLITS=4 under stride-partition (unchanged combine BLOCK_D=128). Smaller split count reduces atomic-barrier contention and combine-phase work while stride-partition's load-balancing remains effective. Not in the "Do not try" list from exp_30/plan.md.
