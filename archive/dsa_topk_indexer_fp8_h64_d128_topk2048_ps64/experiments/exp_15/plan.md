# Plan — exp 15

## Diagnosis

We are in a **local minimum around exp 10 (0.047 ms mean)**. Four
consecutive reverts (exp 11-14) tried micro-knobs on score_kernel
(num_warps), remap_kernel (BLOCK_K), and `.item()`-sync-based
dynamic K — all either noise or structurally blocked by the
`.item()` barrier established in LESSONS.md. The profile
(exp7 run) says **torch.topk is 45-52 µs = 45% of event total
latency** — the single largest remaining phase, and no experiment
since exp 8 has attacked it. exp 8's attempt used `tl.sort` on a
packed uint64 key; that approach scales badly past BLOCK_N=2048
(82 → 368 µs on a876010b), and the branched fallback tied in A/B
because the small-workload wins were absorbed by Python branch
overhead.

**The user's list of unexplored axes points at the same lever from
a different angle: a *selection variant* for small K.** exp 8's
fallback section explicitly listed a "tile-merge approach: load
2048-tile at a time, sort, merge with running top-2048. 3
iterations for max_scored=5696" — never actually built. That is
what this experiment implements.

## Strategy

**Pivot** (structural algorithm change, not a micro-knob).

Replace `torch.topk` with a single Triton `topk_kernel` that uses
a **tile-merge selection** over chunks of BLOCK_CHUNK=2048 scores,
maintaining a running top-2048 per batch in registers. No
BLOCK_N=8192 monolithic sort (which is what made exp 8 blow up).
No CPU sync. Uniform across all workloads — no Python branch.

## Actions (priority ordered)

### 1. **What**: A new Triton `topk_kernel` with grid `(B,)`, one program per batch.

Inside each program, process `scores[pid_b, :]` in successive
chunks of BLOCK_CHUNK=2048 elements. Maintain a running top-2048
`(score, index)` pair set in registers. At each new chunk:

1. Load 2048 new scores + their base offset indices.
2. Concatenate with running top-2048 → 4096 (score, index) pairs.
3. Sort the 4096-pair tile by descending score via `tl.argsort` on a
   packed `(monotone_f32_bits, index)` key (exp 8's monotonization
   trick is re-used here — exact-match tie-breaking is proven).
4. Keep the top-2048 as the new running set; discard the bottom
   2048.

After all chunks processed, the final running top-2048 is the
kernel's answer. Store to `topk_idx[pid_b, :]` as int64.

**Why tile-merge instead of a monolithic sort**:
- `tl.sort/argsort` on BLOCK_N=8192 blows up (exp 8 measurement:
  368 µs on a876010b with BLOCK_N=8192; torch.topk does the same
  in 60 µs). `tl.argsort` on BLOCK_N=2048 is in the safe regime
  per exp 8 LESSONS.md.
- Work per chunk: one 4096-element sort (inside the safe regime,
  since 4096 ≤ 2^13 and exp 8's 4096 data point was "+27%
  regression" vs torch.topk — a regression yes, but only one
  such sort per chunk, and the merge structure means we're only
  sorting the *combined* tile).
  - Correction: exp 8's 4096 data point was with BLOCK_N=4096
    (sort of the full input). A 4096-tile sort that's merge-only
    (input already half-sorted) may be faster in practice.
- For a876010b (B=29, max_num_pages=89, max_scored=5696): 3
  chunks × ~5-8 µs/chunk = ~18-24 µs. Much less than torch.topk's
  51-60 µs.
- For BLOCK_N ≤ 2048 workloads (59/128): 1 chunk, no merge at
  all. Purely a small sort. Exp 8 V1 proved this regime works
  (30cecff1: 22 → 11 µs).

**Why**: This directly attacks the single biggest phase of the
remaining latency and has a known theoretical win shape from exp 8's
V1 (tl.sort was fine at BLOCK_N ≤ 2048). The tile-merge approach
*keeps* BLOCK_N=2048-scale sorts throughout, never exceeding the
safe threshold.

**Impact**: Ceiling is ~30 µs of savings per call on medium/large
workloads (torch.topk 51 µs → triton ~20 µs). Small workloads
preserve exp 8 V1's 50% cut (22 → 11 µs). Rough target: mean
0.047 → 0.035-0.040 ms (-15-25%). Correctness via monotonization
(exact-match proven in exp 8).

### Code sketch (conceptual, no copy-paste)

```python
@triton.jit
def topk_kernel(
    scores_ptr,       # float32 [B, max_scored]
    topk_idx_ptr,     # int64   [B, topk]
    stride_sb, stride_st,
    stride_ib, stride_ik,
    max_scored,
    TOPK: tl.constexpr,         # 2048
    BLOCK_CHUNK: tl.constexpr,  # 2048
    NUM_CHUNKS: tl.constexpr,   # ceil(max_scored_cap / BLOCK_CHUNK)
):
    pid_b = tl.program_id(0)
    running_key = tl.full([TOPK], 0, tl.uint64)  # "top-2048 so far"

    for chunk_id in range(NUM_CHUNKS):
        base = chunk_id * BLOCK_CHUNK
        offs = base + tl.arange(0, BLOCK_CHUNK)
        mask = offs < max_scored
        s = tl.load(scores_ptr + pid_b * stride_sb + offs * stride_st,
                    mask=mask, other=-float('inf'))

        # Pack (monotone_bits, index) into uint64, exp 8-style:
        mono = monotone_bits(s)  # see exp_8/indexer_fused.py
        inv_idx = (MAX_INDEX - offs).to(tl.uint32)
        new_key = (mono.to(tl.uint64) << 32) | inv_idx.to(tl.uint64)

        # Concat running + new (shape [TOPK + BLOCK_CHUNK]) and sort.
        # tl.cat is a compile-time concatenation; if not available, use
        # a stacked load via ranges.
        combined = tl.cat(running_key, new_key)
        sorted_combined = tl.sort(combined, dim=0, descending=True)

        # Keep top TOPK:
        running_key = tl.reshape(tl.gather(sorted_combined, tl.arange(0, TOPK)),
                                 [TOPK])

    # Unpack indices from final running_key:
    sorted_idx = (MAX_INDEX) - (running_key & 0xFFFFFFFF).to(tl.int32)
    tl.store(topk_idx_ptr + pid_b * stride_ib + tl.arange(0, TOPK) * stride_ik,
             sorted_idx.to(tl.int64))
```

- Use `NUM_CHUNKS = triton.cdiv(max_scored, BLOCK_CHUNK)` chosen at
  launch time (as a constexpr for each shape; may need a small
  autotune to cover {1, 2, 3, 4} chunk variants — all compile once).
- Running key carries `(mono_bits, inv_idx)` so the final
  untransformation is just a mask on the low 32 bits.

### 2. **What**: After the topk_kernel succeeds, evaluate whether to FUSE it with `remap_kernel`.

If topk_kernel emits int64 indices into `scores`-space and
remap_kernel maps to token indices, there is a potential fusion:
topk_kernel writes directly to `topk_indices[b, :]` after the
block_table lookup. This saves the `topk_idx` intermediate buffer
(~16 KB per batch × B) and one kernel launch (~7 µs saved).

Gate this as a **separate follow-up experiment** (exp 16+). Only
do it if exp 15 lands a clear A/B win. One change per iteration.

**Why**: Follows the one-change-per-iteration rule. Unclouded A/B.

### 3. **What**: Regression guard — fall back to torch.topk if correctness fails on any workload.

During development, run `--quick` first for compile check, then
stride-8 A/B. If any workload's `matched_ratio` drops below 1.0,
roll back and investigate ties or overflow.

**Why**: Exp 8's monotonization was proven exact-match across 128
workloads. Re-using the same scheme should preserve correctness,
but the tile-merge restructuring is new and could have off-by-one
issues in chunk boundary handling.

## Do not try

- **Monolithic `tl.sort` / `tl.argsort` at BLOCK_N=8192** (exp 8:
  368 µs on a876010b, +349% regression). The tile-merge variant
  exists precisely to avoid this.
- **Monolithic `tl.sort` at BLOCK_N=4096** (exp 8: +27% regression
  vs torch.topk). Sorting the merge-concatenation is a different
  shape of work (already half-sorted input in one half); this may
  behave better, but do NOT attempt BLOCK_N>2048 outside the
  tile-merge context.
- **Branched fallback "if small use triton else use torch.topk"**
  (exp 8 V2: Python branch overhead absorbed all wins; A/B tied
  6/16). This experiment should be uniform — one kernel path for
  all workloads.
- **num_warps/BLOCK_K micro-tuning on score_kernel or remap_kernel**
  (exp 5, 11, 12 all reverted — proved to be wash or regression).
- **`.item()`-based dynamic K** (exp 13, 14 reverted — structural
  barrier per LESSONS.md). The new topk_kernel should consume
  `TOPK=2048` as a compile-time constant.
- **PAGES_PER_PROGRAM > 1 on score_kernel** (exp 3: regressed
  large workloads +85%). This experiment does not touch
  score_kernel.
- **Radix-select** (hinted at in exp 8 next steps). More complex
  than tile-merge, would require atomic counters and 2-3 passes.
  Save for exp 16+ if tile-merge succeeds but plateaus before the
  ceiling.
- **Pre-filling scores with `-1e30` to allow score_kernel's
  early-return to omit the sentinel store**: the math doesn't
  pan out (pre-fill launch ~2-3 µs > sentinel-write savings ~0.1 µs
  at HBM bandwidth; see diagnosis notes).
- **Q broadcast-load via grid axis swap or persistent kernel**: Q
  is already reused via L2 at high hit rate on B200 (Q total
  ≤232 KB fits easily); exp 3 exhausted this axis.

## Success criterion

- **Correctness**: 128/128 exact match (`matched_ratio = 1.0`).
- **A/B vs exp 10** (stride 8, paired same-VM): B wins ≥ 12/16,
  mean Δ ≤ -5% (-0.0025 ms).
- **Full-run**: mean ≤ 0.043 ms (from 0.047). Stretch target: ≤
  0.038 ms.
- **Per-workload profile**: torch.topk event slice disappears
  from the event breakdown (becomes 0, replaced by ~15-25 µs of
  topk_kernel time).

## Coordination notes

- **Iterate via `--quick` first** to get a compile + correctness
  signal in ~2 min. Then `--stride 8` A/B via
  `scripts/ab_benchmark.py` for perf signal.
- **Read `exp_8/indexer_fused.py` for the monotonization code** —
  it's proven correct (packed `(mono_bits.u64 << 32) | inv_idx.u64`
  descending sort = exact torch.topk match). Copy that exactly; do
  not re-derive.
- **Triton `tl.cat` is available in recent Triton**; if not, the
  alternative is a two-buffer merge-sort step where the running set
  is reloaded from a small fp32 intermediate. Test with `tl.cat`
  first; fall back to the two-buffer variant only if `tl.cat`
  fails to compile.
- **Do not combine with remap_kernel fusion in this experiment**.
  Keep topk_kernel emitting int64 indices into score-space; let
  remap_kernel map them as before. Fusion is a follow-up.
- **If NUM_CHUNKS-as-constexpr triggers many compilations**,
  quantize to {1, 2, 3, 4} discrete variants via an autotune key
  derived from `triton.cdiv(max_scored, 2048)`. This avoids
  unbounded JIT cache growth.
