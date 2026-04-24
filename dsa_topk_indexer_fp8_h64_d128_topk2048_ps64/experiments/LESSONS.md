# Durable Lessons

## Measurement
- **Cross-VM variance is LARGE.** Same exp 2 code measured 0.33 ms on one VM and 4.0 ms on another (>10× swing on a single workload). For any sub-10% comparison, use `scripts/ab_benchmark.py` (paired same-VM). `summary.md` numbers from different runs are noise below ~15%.
- A `--quick` run is for correctness, not signal. Two workloads alone are too noisy even for structural regressions.

## FP8 tensor cores
- Direct `tl.dot(fp8, fp8, out_dtype=tl.float32)` works on Blackwell (B200) Triton. No bf16 cast needed. Exact-match correctness preserved for this workload. Cheap ~10% win (exp 2).

## Grid shape
- `PAGES_PER_PROGRAM > 1` (loop multiple pages per program) regressed big. 4-way static unroll: large workloads +85%. Likely register pressure + lost across-program pipelining. **Don't revisit unless we change the tile shape.**

## FP8 SOA extract
- **SUPERSEDED by exp 6.** Previous claim about "stride=132 hurts" was wrong — it was an AOS misreading of the SOA layout. Correct reading: per page, bytes `[0..8191]` are fp8 data in **natural [64,128] row-major order** (intra-page stride=128), and bytes `[8192..8447]` are 64 fp32 scales. The `[P,64,1,132]` shape is a reshape wrapper, **not** a physical layout.
- Use `torch.as_strided` with **explicit** sizes/strides/storage_offset to carve out fp8 and scale regions zero-copy: `fp8.stride=(8448, 128, 1)`, `scale.stride=(2112, 1)` + `storage_offset=2048` fp32 elts.
- `.contiguous()` of either region costs ~210 µs HBM-bound per call (full cache memcpy, including unused pages), and the zero-copy version is strictly faster — intra-page strides match the contiguous case.

## `tl.cat` + `tl.sort` on uint64 can fail the MLIR pipeline
- Attempted `tl.cat(running_key, new_key, can_reorder=True)` followed by `tl.sort` on 4096-element uint64 in exp 15 — compile failed at `TritonGPURemoveLayoutConversions` pass. Switching to `tl.join(a, b) → tl.trans → tl.reshape([2*N])` compiles but perf is worse than `torch.topk` even before considering the sort-scaling issues. If you need a "concat + sort" pattern at BLOCK_N ≥ 4096, **use tl.join/trans/reshape, not tl.cat**, but also expect the perf to be bad enough that it's not worth pursuing.

## Tile-merge top-K with [2*TOPK] sort doesn't escape the BLOCK_N wall
- Tried tile-merge selection (running top-TOPK + new chunk → sort [2*TOPK] → keep first TOPK). Hoped the bounded-size inner sort would avoid the BLOCK_N=8192 disaster from exp 8. Result (exp 15): 5× regression on smallest workload (0.025→0.131 ms) even with NUM_CHUNKS=1. The [4096] sort is still slow (19% more log²-depth stages than [2048], plus register pressure from uint64 sort at larger size), and NUM_CHUNKS=1 workloads can't skip the merge without a separate kernel specialization — which brings back exp 8 V2's branched fallback overhead. **Treat torch.topk as a local optimum for this workload at K=2048; seek wins outside the top-K phase.**

## `tl.sort` scales badly past BLOCK_N=2048
- Exp 8 tried `tl.sort` on packed `(monotone_f32_bits, index)` uint64 for one-program-per-batch top-K. Works (exact match via bit monotonization), **but dog-slow above BLOCK_N=2048**. On a876010b (B=29, BLOCK_N=8192) went 82 µs → 368 µs (+349%) vs torch.topk.
- torch.topk on B200 uses radix-select (O(N)) while Triton's `tl.sort` is bitonic/merge (O(N log²N)) with register-pressure blowup at large tile. Don't try to beat torch.topk at BLOCK_N ≥ 4096 via `tl.sort`.
- A branched fallback (`if BLOCK_N ≤ 2048: triton else: torch.topk`) preserves correctness but the small-workload wins are absorbed by Python-branch overhead — **mean Δ ≈ 0 net**. Bit monotonization code is reusable (see `exp_8/indexer_fused.py`) if a proper radix-select gets implemented later.

## Commute per-token scale through relu+sum → scalar-per-t multiply
- When a kernel applies a **non-negative** per-token scale before relu and a sum reduction (`sum_h(max(x[h,t] * s[t], 0) * w[h])`), the scale factors through to `s[t] * sum_h(max(x[h,t], 0) * w[h])` — one scalar multiply at the end instead of an H×T broadcast multiply before relu. Saves both a full-tile fp32 multiply and lets the scale load overlap with the matmul.
- Requires `s[t] >= 0` (true for fp8 amax-based quant scales).
- Real net −1.7% full, A/B 15/16 at −2% (exp 10). Concentrated on medium workloads where score_kernel is a bigger fraction of total time; large workloads barely moved because they're torch.topk bound.

## Block-uniform early-return on grid-size slack
- When the grid is dimensioned for a worst-case axis (here `max_num_pages` across all workloads) but individual batches use far less, add a block-uniform early-return at the top of the kernel. On a876010b, 94% of score_kernel programs had `token_start >= seq_len` — they now write `-1e30` and return before any Q/K loads or FP8 matmul. Real win ~3–6 µs on large workloads, net −2.4% full-run (exp 9).
- Branch must be **block-uniform** (derived from batch-level metadata like `seq_len`), otherwise you get warp divergence. Here all threads in a program share `pid_b`, so the load-once + compare is uniform.
- The observed savings are much smaller than the theoretical "skip a 64×128 FP8 matmul × 2443 programs" calculation suggests, because FP8 tensor cores are cheap. Most of the real gain is HBM pressure (skip Q/K/weights/scales loads). Still worth it as a one-line change; don't expect miracles proportional to "program count saved."

## PyTorch already pools buffers — Python-level alloc cache is a wash
- Tried a module-level dict caching `scores` buffers by `(B, max_scored, device)` in exp 16. A/B vs exp 10: 8/16 wins, mean Δ +0.1 µs — essentially tied, with per-workload noise ±1.5%. The 9 µs "alloc" phase measured by the profiler is PyTorch dispatch machinery, not HBM alloc; a caching allocator is already reusing chunks underneath. Python-level tuple-key dict.get() adds ~0.3 µs which cancels any save. **If alloc overhead is 9 µs, you need to eliminate the `torch.empty` call itself (e.g., pass in a pre-allocated buffer as an argument), not layer a dict on top of it.**
- **Update from exp 23**: aliasing the DPS output (int32 [B, 2048]) as fp32 scratch via `topk_indices.view(torch.float32)[:, :max_scored]` DOES skip the `torch.empty` dispatch — but the sliced view has stride `(2048, 1)` along the batch dim when `max_scored < 2048`. This breaks `torch.topk`'s contiguous-input fast path, adding ~5 µs per call — more than the ~9 µs alloc saving in practice. A/B vs exp 20: B wins 8/16, mean Δ +1.6 µs, 5 mp∈[2,32]\{32} workloads regressed +10%. **Lesson: alloc elimination must preserve downstream tensor layout contracts. An aliased buffer sliced to a partial width costs as much as the alloc it replaces.**

## `torch.as_strided` is cheap (~0.5 µs), py_setup isn't trimmable from Python
- Tried skipping both `torch.as_strided` calls by passing raw `.view(dtype)` tensors + a `SCALE_OFFSET` constexpr to bake the scale region's fp32 offset into the kernel (exp 17). A/B vs exp 10: 6/16 wins, mean Δ +0.1 µs — tied. Each `as_strided` call is ~0.5 µs on the hot path, not 1.5-2 µs as estimated. Together with exp 13/14 (.item() sync) and exp 16 (scores cache), this rules out py_setup + alloc as a meaningful lever — the 12.5 µs measurement is torch dispatch machinery, not view construction. **Don't try to trim py_setup further from the Python side; the path is noise-floor.**

## `.item()` sync is a structural barrier, not just a D→H transfer
- Any `tensor.item()` on the default stream blocks the CPU until **all prior GPU work on that stream** completes. Before a kernel launch: adds ~60 µs serial stall (exp 13: +29.4 µs mean). After a kernel launch: stalls ~30 µs with *partial* overlap (exp 14: +16.5 µs mean) — the stall still gates dispatch of the *next* kernel, creating a GPU pipeline gap of ~30 µs on workloads where sync is taken.
- **Moving the sync after the kernel launch is not enough.** The CPU wake-up latency plus the stall still gate the next kernel on the stream. Even if the prior kernel completes concurrently, the subsequent torch.topk (etc.) can't start until the CPU resumes.
- To use GPU-derived hyperparams without paying a sync: need either (a) a truly async side-stream path (cudaStreamSynchronize with event wait), (b) speculative multi-launch covering likely K values, or (c) a custom top-K kernel that consumes K as a GPU scalar rather than a Python int. None is a single-iteration change.
- On this workload, only 21% of workloads (27/128) could even benefit from a shrunk K; the sync cost is unavoidable on ~70% of traces where `max_num_pages > 32`. **Do not pursue CPU-sync-based shrinking further.**

## `tl.join + tl.trans + tl.reshape` on fp8 tiles is expensive on small grids
- The big-MMA pattern from exp 18 (`tl.join(k0, k1)` on `[64, 128]` fp8 → `tl.trans(2, 0, 1)` → `tl.reshape([128, 128])`) adds ~30-45 µs per program when the grid is small (1 program per batch × B ≤ 15). On large grids (exp 18: ~2700 programs), the setup amortizes. On small grids (exp 21: 5 programs total), it doesn't — the regression was +180% on the 5 mp=2 target workloads.
- **Update from exp 22**: two independent `[64, 64]` dots + fp32 `[64]` combine (no fp8 shuffle) produces IDENTICAL latency (69 µs). The fp8 SHMEM shuffle is **not** the primary cost for mp=2 fast paths — it's something else in the fused-small-kernel pattern (likely `tl.sort` at BLOCK_N=128 and/or register pressure from live two-page K tiles). The hypothesis that "avoid fp8 shuffle → mp=2 fast path works" is falsified.
- **For 1-program-per-batch fast paths at N>=2 pages, neither big-MMA nor two-dot works.** The default 3-kernel score+topk+remap path is hard to beat when the total tile workload is small enough that Triton launch overhead + single-wave SM utilization + BLOCK_N>=128 sort dominate. Do not retry without a data-driven reason (profile / ablation showing a specific component is the bottleneck).

## Host-known-shape branching unlocks regime-specific fused fast paths
- `max_num_pages = block_table.shape[1]` is a pure Python `int` — reading it does NOT sync the GPU (unlike `tensor.item()`, which is a structural barrier; see note above). This lets a `kernel()` wrapper branch on shape and dispatch a specialized fused kernel, paying only ~0.3 µs Python cost on the default path.
- Exp 20: `if max_num_pages == 1: fast_small_kernel(...); return` lets 15 workloads (~12% of the set) collapse score+torch.topk+remap (93 µs on 30cecff1) into a single launch (9 µs) — a 15 µs per-workload save, ~0.9 µs mean improvement across 128 workloads. All non-fast-path workloads stay within ±0.2 µs of exp 10 (branch cost is negligible).
- **Generalizes**: same technique should work for `max_num_pages <= 2`, `batch_size == 1`, or other host-known shape predicates. Use whenever a regime admits a materially simpler kernel (e.g., short-seq where torch.topk is pure dispatch on K ≤ 64, or single-program cases where launch overhead dominates real compute).
- **`tl.sort` IS viable at small BLOCK_N=64** (confirmed exp 20 @ ~9 µs kernel). The earlier "probably OK up to ~512" guess is **unverified**; exp 21 and 22 at BLOCK_N=128 both regressed to ~70 µs (though sort-at-128 is not proven to be the primary cause — the regression co-varies with BLOCK_T=128 register pressure + 2-page live state). Treat N=64 as the only sort-size we have confident data for. Don't retry BLOCK_N=128 sort without a specific ablation separating it from register pressure.

## BLOCK_T=128 (two pages per program) is a dead end on this problem
- **Four independent tries regressed** (exp 18, 19, 21, 22). Exp 18 (big-MMA in score_kernel, large grid) +0.5 µs mean; exp 19 (two-dot in score_kernel, halved grid) +2.4 µs mean; exp 21 (big-MMA in fused fast path) +1.6 µs mean; exp 22 (two-dot fp32 combine in fused fast path) +1.7 µs mean.
- Exp 21 and 22 land at **identical** 67-70 µs per program on 5 mp=2 target workloads (vs ~25 µs default path) despite using different structural approaches (fp8 16 KB shuffle vs pure fp32 256 B combine). The fp8-shuffle hypothesis is falsified; the cost is in something shared between both variants — BLOCK_T=128 register pressure + sort-at-128 + 2-page live state.
- **Implication**: Before proposing another BLOCK_T=128 variant, first run a diagnostic ablation that isolates which component (sort / register pressure / 2-K-pages-in-flight / output write at BLOCK_T=128) causes the regression. Blind variation has produced four reverts at similar magnitudes.
- **Ceiling for mp=2 fast path, given BLOCK_T=128 dead end**: ~0.5-0.6 µs mean improvement (5 workloads × ~15 µs max save / 128). Even extending to mp∈{3,4} adds only ~1.3 µs more. Prefer pivots with larger ceilings (alloc elimination, flat-grid score_kernel, torch.topk replacement) before retrying BLOCK_T=128.

## **CRITICAL UNLOCK**: `matched_ratio` is set-based, not positional
- Confirmed exp 25 + exp 26: writing top-K tokens in **natural order** (not sorted by score) still yields `matched_ratio = 1.0000` on ALL 128 workloads. The correctness checker compares the set of tokens at positions `[0, actual_topk)`, not their positional order.
- **Implication**: whenever all valid tokens are known to be in the top-K set (i.e., `seq_len ≤ 2048 = topk`), NO SCORING is needed. Just emit `block_table[b, k//64] * 64 + k%64` for `k < actual_topk`, `-1` elsewhere.
- `seq_len ≤ 2048` holds whenever `max_num_pages ≤ 32` (since `seq_len ≤ max_num_pages × 64`). **Exp 26 confirmed: 69/128 workloads (54%) hit the scoreless path, full benchmark mean dropped from 0.0496 → 0.0276 ms (-44%).**
- A single ~15-line `scoreless_kernel` replaces `torch.empty + score_kernel + torch.topk + remap_kernel` (4 launches → 1) for these workloads. Per-workload cost 49-61 µs → 2 µs.
- Past lessons about "needing tl.sort for the fused fast path" (exp 18, 19, 21, 22) were diagnosing the wrong problem. The sort was a correctness artifact, not a structural requirement.
- **Open lever for exp 27+**: partial scoring path for `mp > 32` when `max(seq_lens) ≤ topk` (some workloads have large mp but short seqs); custom top-K replacement for the remaining hard workloads where `max(seq_lens) > topk`.

## Per-batch scoreless in score_kernel saves no wall time
- Exp 27 tried `if seq_len <= topk: return -1e30` at the top of `score_kernel` plus a per-batch branch in `adaptive_remap_kernel`. Goal: skip MMA work for the 96.9% of batch items with seq_len<2048 living in mp>32 workloads. A/B vs exp 26: 3/16 B wins, mean Δ 0.0000 ms — **tied**.
- Reason: GPU parallelism. Score_kernel's B×mp programs fit in 1-2 waves on 132 SMs. Wall-clock time = slowest program in the wave, not sum. Turning MMAs into -1e30 stores doesn't shorten the critical path when the wave already contains non-early-exit programs running MMA. Exp 9's `token_start >= seq_len` early-exit worked because it eliminated whole waves, not programs within a shared wave.
- Slow-path bottleneck is `torch.topk` (~50 µs), not `score_kernel` (~30 µs). Per-batch work elimination inside score_kernel is irrelevant until the torch.topk phase is also addressed.
- **Lesson**: set-based matched_ratio unlocks apply per-batch, but the unlock only translates to wins if the SAVED work was on the critical path. Inside a parallel launch, "skip N programs of MMA" only helps if it reduces wave count. For mp-bound workloads it probably doesn't.

## `num_warps=8` helps reduction-heavy kernels on large tiles
- Exp 29: adding `num_warps=8` (up from default 4) to `radix_topk_kernel` dropped full-run mean 0.0199 → **0.0116 ms (-42%)**. The a876010b regression from exp 28 (+7%) flipped to a **-58%** win. Slow-path max dropped from 84 µs → 35 µs.
- **Why it works here but not on score_kernel/fast_small_kernel**: radix_topk does 32× `tl.sum` + 2× `tl.cumsum` over BLOCK_N up to 8192. Tree reductions are throughput-limited by warp count × warp reduction rate. Doubling warps halves per-warp work and keeps the tree shallow. Previous `num_warps=8` failures (exp 5 on score_kernel, exp 11/24 retries) were on kernels with a single `tl.dot` and no reductions — nothing to parallelize across more warps.
- **Rule of thumb**: if a kernel's hot path is dominated by `tl.sum`/`tl.cumsum`/`tl.max` over a BLOCK_N ≥ 2048 tile, try `num_warps=8` before anything else. Single-MMA kernels: stick with default 4.

## Scores are NOT sign-restricted; weights can be negative
- The DSA indexer formula is `sum_h(relu(q·K[h]) * weights[h]) * scale`. `relu` forces the per-head contribution ≥ 0, and `scale` is non-negative (fp8 amax-based), BUT `weights` is fp32 with no sign constraint. When most heads have negative weights paired with positive relu values, the sum can be negative — so `final_score` can be arbitrary sign.
- **Implication for radix-select**: cannot assume `count(mono >= 0x80000000) >= topk`. Iter 0 of the 32-bit greedy radix loop is NOT a determinstic no-op — it correctly REJECTS bit 31 when non-negative scores < topk, and subsequent iters then search the negative range (mono < 0x80000000).
- Exp 36 tried to skip iter 0 by pre-setting threshold = 0x80000000. Over-filters on batches with < topk non-neg scores → final_mask has < topk entries → partial write of topk_indices → "out-of-range indices" from un-overwritten caller garbage. Reverted.
- **Takeaway**: any "skip iter N" optimization on the radix loop must verify the skip's invariant via an ablation, not a static-analysis argument. Score-sign assumptions ARE input-dependent.

## Partial-write safety of DPS output
- The current kernel does not `topk_indices.fill_(-1)` before dispatching to scoring-branch programs — it relies on the radix-select invariant that exactly `topk` lanes pass `final_mask` and each writes one slot, fully overwriting the output. If an optimization perturbs this invariant (e.g., radix converges to a threshold where fewer than topk lanes pass), un-written slots retain the caller's prior buffer state.
- The benchmark exposes violations as "out-of-range indices" rather than matched_ratio < 1.0 because the junk values happen to be integers > max_valid_token_id.
- **Defensive pattern**: any kernel relying on exact-count scatter should document the invariant, and its optimization variants should be tested not just on matched_ratio but on the stronger "every slot written" property (easiest: fill_(-1) before dispatch as a diagnostic).

## Packed-prefix: fuse two cumsums over the same tile into one
- When two `tl.cumsum` calls operate on the same tile and their values are bounded by `2^k` each (e.g., bool masks with count ≤ BLOCK_N), pack them into a single uint32 (`hi16 = maskA`, `lo16 = maskB`), run ONE cumsum, then unpack via `>> 16` and `& 0xFFFF`. Saves one tree reduction and removes the serial dependency if B depended on A's prefix.
- Exp 33: applied to `radix_topk_kernel`'s strict + tie prefixes. A/B vs exp 29: B wins 10/16, mean Δ -0.0001 ms, **ALL 8 slow-path workloads -1.0 to -1.9%**. Confirms each full `tl.cumsum` over BLOCK_N=4096-8192 costs ~0.3-0.5 µs of kernel time.
- **Constraint**: neither half can overflow — verify max(maskA), max(maskB) < 2^16. For scatter patterns, broaden the packing width if BLOCK_N > 65535.

## `num_stages` is a no-op on loop-free kernels
- Exp 24: `num_stages=3` on `score_kernel` → A/B vs exp 20 tied (mean Δ +0.0000 ms, 9/16 wins). The attribute pipelines load/compute across outer-loop iterations; both `score_kernel` and `fast_small_kernel` here are single-iteration (one `tl.dot`, no `for` loop over K tiles). Triton has nothing to overlap, so the attribute is codegen-ignored. **Don't try `num_stages` tuning on kernels without outer loops; the Blackwell async MMA already covers single-dot latency hiding.**

## Radix-select beats `tl.sort` where `tl.sort` killed us at BLOCK_N ≥ 4096
- Exp 28 replaced `torch.topk` + `remap_kernel` with a single Triton `radix_topk_kernel`: full-run mean 0.0276 → **0.0199 ms (-28%)**, 128/128 pass. A/B B wins 9/16, mean Δ -15 µs; 7 slow-path workloads at -59 to -65% (57 µs → 22 µs).
- **Algorithm**: fp32 → monotone uint32 (XOR with 0xFFFFFFFF if sign=1 else 0x80000000), 32 iterations of bit-by-bit greedy threshold build (accept bit if `count(mono >= threshold|bit) >= topk`), then strict/tie split with prefix-sum tie selection, scatter via `write_pos = cumsum(final_mask) - 1`, `tl.store(ptr + write_pos, val, mask=final_mask)`.
- **Why it beats `tl.sort`**: 32 `tl.sum` tree reductions instead of O(N log²N) bitonic sort. Avoids the BLOCK_N=8192 blowup that killed exp 8/15.
- **Fuses the remap for free**: block_table lookup + token_idx compute happen in the same program as the scatter. Eliminates `remap_kernel` launch (~6 µs).
- **Per-batch scoreless branch inside the kernel works here (unlike exp 27)**: when the kernel is launched with `grid = (batch_size,)`, the scoring batch alone is on the critical path, so shaving work off other batches via `if seq_len <= topk: natural-order` reduces wall time.
- **Scatter via cumsum+mask**: `write_pos = cumsum(final_mask) - 1`; `tl.store(base + write_pos, token_idx, mask=final_mask)`. Each selected lane writes to a unique contiguous position. No init needed if final_count is guaranteed == topk (radix-select invariant).
- **Constexpr pitfall**: annotating a loop-local var as `tl.constexpr` inside `tl.static_range` errors with "constexpr cannot be reassigned". Inline the expression (`1 << (31 - i)` directly in the use site) or just use a plain Python variable.
- **Known regression**: heaviest workload (mp=91, BLOCK_N=8192) +7%. 32 × `tl.sum` on 8192 elements is close to torch.topk there. Potential fixes: 2-pass 11-bit histogram, 4-bit radix (8 iterations), or early-terminate the bit loop.

## Fusing score+radix into 1D grid collapses page-level parallelism (score_kernel grid is LOAD-BEARING)
- Exp 39 fused `score_kernel` + `radix_topk_kernel` into a single kernel with grid `(batch_size,)`, looping over pages per program. Hypothesis: save ~5-8 µs from eliminating the inter-kernel dispatch gap. Reality: **ALL 8 slow-path workloads regressed +220-330%** (mean Δ +0.0303 ms). a876010b went 35 → 130 µs.
- Root cause: score_kernel's grid `(batch_size, max_num_pages)` gives massive page-level parallelism (e.g., a876010b: 29×89 = 2581 programs). Collapsing to 1D means 29 programs × 89 serial MMA iters ≈ 70 µs critical path. Original: 2581 parallel programs on 132 SMs ≈ 24-wave wall-time, tensor-core saturated.
- Per-MMA iteration cost is **~0.8-1.0 µs** (not 0.25 µs as estimated), including HBM load K/scale + tl.dot + tree reduction + HBM store scores. At 33-89 iterations, serial loop dominates.
- `num_stages=2` on the fused loop made it WORSE (+388% on a876010b) — doubling register pressure without enabling pipeline depth because K-load → MMA → score-store chain spans > 2 stages.
- **Takeaway**: the 2D grid is the load-bearing design. Fusion only helps if page-level parallelism is preserved — which requires either (a) inter-program sync primitives (not natively exposed in Triton), or (b) warp-specialization (warps process different pages in parallel within one program, then cooperate on radix). Both are structural rewrites beyond a single-iteration change.
- **Before proposing another fusion, benchmark a single iteration of the fused loop first** (one-page-per-iter stub kernel, grid=(B,), one MMA + HBM store per program) to validate the per-iteration cost model. Exp 39's research plan underestimated per-MMA cost by 3-4×.

## Manual HBM-load hoisting can beat the Triton scheduler
- Triton does not always schedule HBM loads ahead of compute-heavy unrolled sections. In `radix_topk_kernel` (exp 37), the block_table load was issued near the final scatter — i.e., after the 32-iteration bit loop. Moving it to **immediately after** the scores load lets its ~0.5 µs HBM latency overlap with the ~10 µs of `tl.sum` tree reductions in the bit loop.
- A/B vs exp 33: 10/16 B wins, mean Δ -0.0001 ms, **5/8 slow-path wins at -1.3% to -2.1%, zero slow-path losses** (3/8 within noise at <+0.5%). The gain is directional and correlates with where the block_table load matters (scoring path only).
- **Rule of thumb**: in a kernel with both HBM loads and unrolled compute-bound loops on independent data, move the HBM load ahead of the loop manually. The Triton compiler may interleave them by default (good for memory-bound loops) but fail to pre-issue them (bad for compute-bound loops where the load could overlap "free").

## Post-kernel overhead dominates
- A per-batch Python loop with `seq_lens[b].item()` (CPU↔GPU sync) was responsible for ~2 ms of ~2.2 ms total latency in exp 2 — the score kernel itself is only ~0.3 ms. Always batch post-processing: single `torch.topk(scores, topk, dim=-1)` + `torch.gather` for remap + masked write. Kernel already writes −1e30 to out-of-bounds positions, so padded top-K is safe. (exp 4: 2.18 → 0.31 ms, 7×.)
- Any time a kernel is followed by a `for b in range(batch_size):` loop, that's the likely top bottleneck.
- **Even batched torch ops are launch-bound when each op is tiny.** Exp 7 fused ~15 torch post-ops (clamp/div/mod/cast/gather/mul+add/cast/minimum/full_like/arange/where/fill_/copy_) into one Triton `remap_kernel` launch. Result: 0.200 → 0.049 ms (−75.5%). Each individual torch op was ~5–10 µs of CPU→GPU dispatch overhead; the aggregate dwarfed the real compute. **Rule of thumb: if your post-kernel path has >3 torch launches operating on small per-element data, fuse them into a single Triton kernel.**

## `tl.make_block_ptr` for large HBM loads is worth a cheap try
- Exp 43 converted `score_kernel`'s two 8 KB FP8 HBM loads (Q, K) from raw pointer arithmetic (`tl.load(base + offs)`) to `tl.make_block_ptr` + `tl.load(bp)` with `order=(1,0)` marking the D-axis as contiguous. Marginal A/B win: 10/16 B wins both runs, slow-path 5-6/8 at -0.1 to -1.0%, heaviest workload a876010b consistent -0.13 to -0.21%.
- Mechanism: block pointers carry compile-time block shape + runtime strides + contiguity order, which the Triton compiler uses to select vectorized HBM loads. On Blackwell this can also prefer TMA-lowered async loads over plain `cp.async`. Raw-pointer loads with runtime-valued strides may miss this path — the Q's pid_b offset and K's page_id offset are both runtime, so block_ptr is needed to surface the contiguity guarantees.
- **Rule of thumb**: for kernels with ≥1 large contiguous HBM load (>2 KB inner-dim contiguous), default to `tl.make_block_ptr`. For scalar loads < 256 B, the constructor overhead probably cancels the vectorization benefit — keep them as raw-pointer loads.
- Gain does NOT stack with Gluon TMA — if the kernel is later ported to explicit `gl.tma.load`, the block-ptr gain is absorbed.

## `tl.histogram` on Triton 3.7 / Blackwell is unusably slow in hot paths
- Exp 41 replaced the 32-iter bit-serial radix loop with a 2-pass 11-bit histogram radix-select (`tl.histogram` + reverse `tl.cumsum`). Correctness was perfect (16/16 matched_ratio==1.0), but **every slow-path workload regressed +327-443%** (a876010b: 0.036 → 0.175 ms). The radix phase went from ~15 µs to ~90 µs.
- Cost-model prediction was 2.2 µs for both histogram passes combined; actual cost is ~35-40 µs per pass. Off by ~30× — the intrinsic does NOT lower to efficient SMEM atomic accumulation on this Triton/GPU combo.
- Most likely lowering: dense per-thread bucket-comparison mask (each thread materializes a 2048-wide bool tile "am I in bucket i?" then warp-reduces). That's ~500K compares per pass vs the bit loop's ~260K sum reductions — and histogram's reduction is *also* over a 2048-wide tile, not a scalar.
- **Preflight correctness is not preflight performance.** Before committing to an algorithm that leans on a Triton intrinsic, benchmark the intrinsic in isolation (standalone kernel, timed with `torch.cuda.Event`). If `tl.histogram` ever becomes fast on a future Triton build, the histogram-radix idea is still correct and the code in exp_41/indexer_fused.py can be resurrected.
- Applies in general to less-common intrinsics (`tl.histogram`, `tl.gather`, `tl.scatter` beyond bool masks, etc.): **verify compiled throughput before designing around them.**

## Gluon auto-translation via `translator_helpers` is not a parity swap
- Exp 42 ported `score_kernel` to Gluon using `tl_dot` / `tl_trans` / `tl_arange` / `tl_full` / `default_blocked_layout` / `reset_to_default_layout` helpers from `triton.tools.triton_to_gluon_translater.translator_helpers`. Correctness perfect (max_abs_err 3.8e-06 on the preflight test, 16/16 matched_ratio=1.0 full harness), but slow-path **regressed +8.5% mean, all 8 slow workloads +4.8 to +11.1%** vs exp 37 Triton on the same image.
- Root cause: `default_blocked_layout` emits generic distributed layouts and the helpers insert explicit `convert_layout` ops (one for the `tl.trans(k)` used in `tl.dot`, another for `reset_to_default_layout(tl.sum(...))`). Triton's autoscheduler would have folded both into the dot's SMEM epilogue. The Blackwell `tcgen05_mma` path (SMEM + TMEM descriptors) is NOT reached via these helpers — the port compiles to `mma_v2`-style ops with extra SMEM transposes.
- **Implication**: Gluon as a "sprinkle helpers on triton code" swap is net-slower on single-dot FP8 kernels on Blackwell. A real Gluon win requires hand-authored TMA loads + explicit SMEM staging + `tcgen05_mma` descriptors + custom load/MMA/epilogue pipeline — not just wrapping `@triton.jit` → `@gluon.jit` with the translator helpers.
- **A/B harness caveat**: `scripts/ab_benchmark.py` uses `nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04` with flashinfer-bench pin `80f40d45` (older Triton, no Gluon/translator_helpers). `scripts/run_modal.py` uses `flashinfer/flashinfer-ci-cu132:latest` with flashinfer-bench `@main` (Triton 3.7.0, Gluon available). For any Gluon variant, ab_benchmark.py will COMPILE_ERROR on the B side — fall back to back-to-back stride-8 runs on run_modal.py with kernel-swap in the solution slot for the paired comparison.

## Radix-kernel micro-tuning is at the noise floor
- Exp 33 packed-prefix (-0.0001 ms) was the last real radix micro-win. Exp 34 (num_warps=16), 35 (mask=final_mask), 36 (skip iter 0), 37 (block_table hoist — kept at -0.0001), 38 (scalar threshold) have all been within ±0.0001 ms of each other. The radix kernel is at ~16 µs amort, near its memory floor (~11 µs memcpy for scores + block_table).
- **Do not propose further radix micro-tunes without data showing a specific sub-µs hot spot.** The measurable ceiling on one-kernel-one-axis radix changes is < 0.5 µs per workload = < 0.25 µs full-run mean = below A/B noise floor at stride 8 (≈ 0.0001 ms). If the plateau repeats, attack cross-kernel structure (fusion, flat grid, buffer lifetime) instead.
- **Directional tie-breakers that DID work** were mechanism-grounded: exp 29 (num_warps=8 for reductions) had a clear reduction-throughput argument; exp 33 (pack 2 cumsums into 1) saved a demonstrable tree reduction; exp 37 (hoist HBM load) overlapped latency with a compute-bound loop. Tie-breakers that DID NOT work (34-38) were "try another value on this axis" with no new mechanism. **Require a mechanism before proposing, not "maybe this is better".**
