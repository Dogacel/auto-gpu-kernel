# Experiment 7 — 2026-04-16

**Description:** Replaced the ~15-op torch post-kernel path (clamp,
div, mod, cast, gather, mul+add, cast, minimum, full_like, arange,
where, fill_, copy_) with a single Triton `remap_kernel`. Only
`torch.topk` remains outside the kernel.

## Results
- Pass: 128/128 (exact match)
- Kernel latency (ms): min 0.022 / mean 0.049 / median 0.049 / max 0.082
- Max abs err: 0.00e+00  |  Max rel err: 0.00e+00
- Mode: full (128 workloads)

## A/B vs exp 6 (same-VM, stride 8)
- B wins 16/16
- mean Δ = −0.1285 ms (−70% avg per workload)
- per-workload range: −53% to −85%

## Delta vs exp 6 (full)
- exp 6 mean = 0.200 ms → exp 7 mean = 0.049 ms (−75.5%, 4× faster)
- max: 0.212 → 0.082 ms (−61%)
- min: 0.188 → 0.022 ms (−88%)

## Cumulative vs exp 2 (FP8 dot baseline)
- exp 2: 2.179 ms mean
- exp 7: 0.049 ms mean
- **~44× speedup** across 5 iterations (4 successful + 1 reverted).

## Kernel design
- Grid `(B, ceil(topk / BLOCK_K))` with `BLOCK_K = 256`. B=32 gives
  256 programs; B=1 gives 8 (acceptable — programs are tiny).
- Per-program work: load 256 int64 topk indices → compute
  `(page_idx, offset)` → clamp page_idx → gather `block_table[b, p]`
  → compute `global_page * page_size + offset` → mask positions beyond
  `min(effective_topk, seq_len[b])` → store int32 with full
  `k < topk` mask (so untouched tail positions also get `-1`).
- Everything previously done via `topk_indices.fill_(-1)` + slice-assign
  is handled by the kernel's store mask + `-1` elsewhere. Drops the
  `fill_` launch too.

## Learnings
- **Launch overhead was the remaining bulk.** Profile attributed
  ~120 µs to topk + remap + mask_write. After this experiment, we're
  at 49 µs total. Even assuming torch.topk alone costs ~30 µs, that
  leaves ~20 µs for everything else — score kernel + remap kernel +
  torch.topk dispatch. The 100 µs gap relative to profile expectations
  is pure launch overhead the previous path was paying for every
  elementwise op.
- **Fusing 15 ops into 1 Triton kernel is an order-of-magnitude win**
  on small-fast workloads (e.g. `30cecff1` went from 184 µs → 27 µs,
  −85%) because those workloads had no GPU compute to hide launch
  latency behind.
- Large workloads saved less in absolute µs (`a876010b`: 188 → 88 µs,
  −53%) because the score kernel still has real compute; but even
  there the post-kernel overhead cut cleanly.

## Remaining costs (approximate)
- `torch.topk(scores, 2048, dim=-1)` — probably the largest single
  piece now. Reduces [B, max_scored] → [B, 2048].
- `score_kernel` — was ~15 µs in the stale profile, unchanged.
- `remap_kernel` — very small but not zero.
- Python → CUDA dispatch for `torch.topk` and the two Triton launches.

Profile will need to be re-run post-exp-7 (old profile is stale;
setup is already eliminated and post-kernel is now 1 launch).

## Next candidates
- Fuse `torch.topk` into Triton (kernel-side top-K with partial sort +
  reduction). This eliminates the scores buffer round-trip.
- Alternatively, compute partial top-K per page in the score kernel
  (each program keeps its top-64 or so, then global merge).
- For small workloads, could specialize: if `seq_len < topk`, skip
  topk entirely and directly emit the range `[0, seq_len)` as token
  indices. Current path still runs full topk for tiny seqs.
