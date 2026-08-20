# Experiment 40 — 2026-04-17

**Description:** Gluon stage-1 pivot per `plan.md`. Ported exp_22's Gluon split scaffold into `solution/triton/sparse_fused.py` and replaced `gl.dot_fma` with Blackwell `bw.tcgen05_mma` for Q@Kc^T and Q@Kp^T. Kept exp_37's `_fused_attn_kernel` (T≤2) untouched. Two-launch pattern (split → combine) per exp_22. Delegated to fresh sub-agent with clean context. Probe-first: wrote `scripts/probe_gluon19.py` to validate API (abs_err 5.3e-5 on toy [HM=128, N=128, D=512]).

## Results
- Pass: 12/12 (quick 2/2, stride-2 12/12) — correctness CONFIRMED on Gluon MMA path
- Kernel latency (ms, stride-2): small=0.005-0.019 (T≤2, Triton path, unchanged) / **large=0.094-0.096 (T≥3, Gluon MMA path)**
- Mode: stride-2
- **Reverted** — 6.1× slower than exp_37 baseline (~0.0155 ms), far exceeds 0.025 ms acceptance gate (3.8× over)

## Root-cause analysis (from sub-agent's forensic)

1. **SMEM pressure forces per-D-chunk Q_nope HBM reload.** Persistent `[HM=128, D_CKV=512]` Q_nope SMEM = 128 KB; combined with Q_pe (16 KB) + kc chunk (16 KB) + kp (8 KB) + scratch fp32 HM×BLOCK_N (32 KB) + compiler padding (~32 KB) ≈ 232+ KB, blew through the 232 KB B200 SMEM budget. Workaround: shrink Q_nope SMEM to `[HM, D_CHUNK=128]` = 32 KB and reload from HBM per D-chunk. Each reload = 128×128×2 = 32 KB × 4 chunks × BLOCK_N-iters × 8 splits → **~128 MB/call extra HBM traffic**. At 8 TB/s ≈ 16 µs overhead.

2. **Per-MMA fence/mbarrier overhead.** HM=128 forces D-chunking into 4×D_CHUNK=128 MMAs for Q_nope plus 1 MMA for Q_pe. 5 MMAs per outer-iter × 4 outer-iters/split × 8 splits = 160 MMA commits per call. Each commit incurs `tcgen05_commit + mbarrier.wait` fence. At ~0.5 µs/fence ≈ 80 µs — **dominant single cost**.

3. **Two-launch tax ~8 µs.** Split + combine = 2 launches; stage 1 explicitly drops atomic-barrier fusion (that's stage 2 / `bw.mbarrier`).

4. **TMEM→SMEM→register HM→H bridge.** TMEM `slice()` is last-dim only; can't slice rows post-MMA. Must store fp32 logits through scratch SMEM and slice(dim=0) to extract top-16 rows. Per-iter round-trip adds latency.

## Why stage 2 (bw.mbarrier) cannot close the gap

Stage 2 refuses split+combine via `bw.mbarrier`. Saves ~8 µs (launch tax). Stage 3 (`num_ctas=8` cluster) saves ~2 µs barrier + ~15 µs Q replication via cluster-shmem. **Combined stage 2+3 optimistic saving: ~25 µs** → best-case end-of-stage-3 latency = 0.095 − 0.025 = 0.070 ms, **still 4.4× exp_37**. The HM=128 D-chunking × 160 MMA-fences (~80 µs) is the dominant bucket and is **not recoverable by stages 2/3**. The only path to eliminate it is HM=64 (reduces phantom-row padding + may allow persistent Q_nope at [64, 512] = 64 KB). But HM=64 halves MMA tile efficiency and still pays ~50 µs in 80 MMA-fences — projected still 2-3× exp_37.

## Verdict

**Reverted to exp_37 baseline.** Confirmed byte-for-byte match via `diff -q solution/triton/sparse_fused.py experiments/exp_37/sparse_fused.py`. Quick re-run shows T=1: 0.005 ms, T=large: 0.016 ms — exp_37 baseline intact.

**The Gluon `bw.tcgen05_mma` path is structurally uncompetitive for this shape (H=16, D_CKV=512, BLOCK_N≥64 per MMA).** Blackwell tensor-core MMA minimum blockM=64 with our H=16 forces ≥75% phantom-row padding, and HM=128 (the more efficient MMA tile) forces D-chunking that compounds fence overhead to ~80 µs. Triton's `tl.dot` hides this same padding behind its own `wgmma` scheduler with no explicit fence surfaced, giving it a framework-overhead advantage of ~5× on this shape. Gluon migration is closed for stage-1-as-defined.

## Discoveries (Gluon stage-1 post-mortem)

1. **`bw.tcgen05_mma` API validated:** signature is `bw.tcgen05_mma(a_smem, b_smem, tmem_acc, use_acc=bool)`; follow-on `bw.tcgen05_commit(tmem_acc) + mbarrier.wait` for synchronization. `bw.alloc_tmem` (or `allocate_tensor_memory` with `value=gl.zeros(...)`) for TMEM allocator. `bw.tmem_load(tmem, layout=...)` pulls into register tile. Canonical pattern in `triton/tools/triton_to_gluon_translater/translator_helpers.py::tl_dot_blackwell`. Validated probe at `scripts/probe_gluon19.py`.

2. **TMEM slice semantics:** `tmem.slice(start, len, dim=last)` — last-dim only. Cannot slice rows post-MMA without an SMEM bridge. Significant constraint for MLA-style H=16-row workloads where MMA forces HM ≥ 64.

3. **HM=128 vs HM=64 trade-off:** HM=128 is the "more efficient MMA tile" per the canonical translator_helpers but forces D-chunking on 512-wide D (SMEM budget). HM=64 would allow persistent Q_nope SMEM at 64 KB but halve MMA throughput. Neither variant closes the 5× gap vs Triton `tl.dot` on H=16 shape.

4. **Stage 1 is not viable ⇒ stages 2/3 are unreachable.** The 3-stage Gluon migration plan (exp_35) was premised on stage 1 matching Triton within 50%. At 6.1× slower, stages 2/3 cannot recover enough to close the gap.

5. **Combine-path Triton is correct.** The exp_22 Triton `_combine_kernel` reduces partial_m/l/acc and produces bit-correct output against the reference. No issue on the combine side; the regression is entirely in the Gluon split-MMA path.

6. **The agent's draft Gluon kernel was lost during revert.** The validated probe patterns (`scripts/probe_gluon19.py`) and the canonical translator reference remain available for any future Gluon attempt. To retry, start fresh from exp_22 scaffold + probe_gluon19.

## Next directions

- **Call research agent.** All cheap Triton axes closed (LESSONS 12, 19, 26, 27, 29, 40, 41, 43, 45). Combine-IO axis closed (exp_28, exp_30, exp_39). Gluon stage-1 now also non-viable (this). Plateau-induced new-angle discovery is warranted — fresh clean-context synthesis needed to find the next move.
- **Candidate angles to probe:** (a) alternative barrier primitives via `tl.inline_asm` (cluster.arrive/wait PTX) — may dodge LESSON-27's planner assert since no tiled ops; (b) warp-specialization inside `_fused_split_combine_kernel` using warp-ID scalar kwargs — never tried; (c) workload-characteristic specialization on T=8 vs T={6,7} — T-specific compile variants with different NUM_SPLITS per T. All three are pure Triton, distinct from prior failed axes.
- **Not fallback:** Another Gluon stage 1 variant (HM=64 + no-chunking) — projected still ~50 µs (3× exp_37), not worth the iteration.
