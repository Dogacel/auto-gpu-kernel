# Experiment 22 — 2026-04-17

**Description:** Fresh-start Gluon rewrite of the T≥3 split path. Goal: get a correctness-passing Gluon kernel in hand to unlock future optimizations (explicit layouts, `bw.mbarrier`, `bw.tcgen05_mma`). T≤2 path kept as plain Triton (`_fused_attn_kernel` unchanged from exp_18). Atomic-barrier fused split+combine dropped; two separate launches (`_split_kernel_gluon` → `_combine_kernel`) used instead.

## Results
- Pass: 12/12 on stride-2 (quick: 2/2)
- Max abs err: 1.56e-02 (T=1, unchanged Triton path) / 7.81e-03 (T≥3, new Gluon path)
- Mode: quick + stride-2

**Latency (stride-2):**
| UUID | T-class | Latency | Path |
|---|---|---|---|
| 0c23b10c | small (T=1) | 0.005 ms | Triton fused (unchanged) |
| b7668cfd | small | 0.005 ms | Triton fused (unchanged) |
| 05f6de65 | small | 0.018 ms | Triton fused (unchanged) |
| e6b849f2 | small | 0.008 ms | Triton fused (unchanged) |
| f77df5ce | small | 0.005 ms | Triton fused (unchanged) |
| 4c46a94b | T=6 | 1.211 ms | **Gluon split + Triton combine** |
| 02d6ae9c | T=8 | 1.288 ms | **Gluon split + Triton combine** |
| 78b2e11c | T=8 | 1.315 ms | **Gluon split + Triton combine** |
| 564007ac | T=8 | 1.530 ms | **Gluon split + Triton combine** |
| 232ed014 | T=8 | 1.277 ms | **Gluon split + Triton combine** |
| 5096e459 | T=8 | 1.544 ms | **Gluon split + Triton combine** |
| 2207f0fd | T=8 | 1.389 ms | **Gluon split + Triton combine** |

Large-T median ~1.30 ms — ~65–85× slower than exp_18 baseline (~0.016 ms). Expected: `gl.dot_fma` is a software FMA (no tensor cores), and layout conversions between fp32 logits/partials go through shared memory. This is iteration 1 of a fresh approach — correctness, not performance, is the gate. **Not a new best; exp_18 remains the best.**

## Design

### Gluon split kernel (`_split_kernel_gluon`)
Each CTA handles `(t, s)` → one split of SPLIT_SIZE=256 indices, BLOCK_N=128 per inner step.

Key layouts (all 8 warps along H rows, warps_per_cta=[8, 1]):
- `qn_layout`: `[1, 8] × [1, 32] × [8, 1]` for `[H=16, D_CKV=512]`
- `qp_layout`: `[1, 2] × [1, 32] × [8, 1]` for `[H=16, D_KPE=64]`
- `pn_layout`: `[1, 4] × [1, 32] × [8, 1]` for `[H=16, BLOCK_N=128]`
- `kc_layout`: `[1, 8] × [16, 2] × [8, 1]` for `[BLOCK_N=128, D_CKV=512]` — rows along threads_per_warp
- `kp_layout`: `[1, 2] × [16, 2] × [8, 1]` for `[BLOCK_N=128, D_KPE=64]`
- `h_layout`: 1D `[32] × [8]` for H=16 reductions (replicated; warps over-cover)

Dots use `gl.dot_fma` with fp32 operands converted to `DotOperandLayout(op, parent, k_width=0)`:
```python
q_nope_f32 = q_nope.to(gl.float32)
kc_t_f32 = kc_t.to(gl.float32)
q_nope_dot = gl.convert_layout(q_nope_f32, dot_q_pn)
kc_t_dot = gl.convert_layout(kc_t_f32, dot_k_pn)
logits = gl.dot_fma(q_nope_dot, kc_t_dot, acc_zero)  # fp32 acc
```

Online softmax is straight-line (no atomic barrier) — write partial m/l/acc to HBM, then launch the combine.

### Triton combine kernel (`_combine_kernel`)
Identical structure to exp_7's D-parallel combine. 8 programs × (T, 8) grid, each reduces NUM_SPLITS=8 partials over 64 channels of D. Kept in plain Triton because:
1. Simple, known-correct reference for bring-up
2. Removes the atomic-barrier variable from the fault hunt
3. Cheap relative to split (~6 µs in exp_18 profile)

## Discoveries (Gluon API on Triton 3.7.0 / sm100)

Eighteen probe scripts in `scripts/probe_gluon*.py`. Durable findings:

1. **`gl.dot_fma` requires `DotOperandLayout(op, parent, k_width=0)` with a `BlockedLayout` parent** — `k_width=0` is mandatory when parent is blocked ("ttg.dot_op kWidth parameter is not supported when the parent is a blocked layout").
2. **`gl.dot_fma` operand dtypes must all match** (and acc dtype must match). Cast bf16 → fp32 before the convert_layout and use fp32 acc. `FMA.cpp aElem.getType() == tgtTy` asserts otherwise.
3. **`warps_per_cta` must sum to `num_warps`** — can't drop warps with a smaller warps_per_cta for small tensors (e.g. H=16). Use replication (same warp covers same row) by over-covering the tile.
4. **1D `gl.arange` layout indexing for 1-D scalars** must be a plain `BlockedLayout([x], [y], [z], [0])` — no SliceLayout wrapper.
5. **`gl.barrier`** is listed in `gl.__init__` but is not resolvable from a `gluon.jit` body. For cross-CTA sync within a single launch, use `gl.atomic_add(ptr, 0, sem="acquire")`.
6. **`gl.convert_layout`** is the bridge between SliceLayout-broadcasted 1-D tensors and their consumers. For example, after `m_local = gl.max(logits, axis=1)` (layout = `SliceLayout(1, pn_layout)`), converting to `h_layout` is needed before scalar-scalar ops with other `h_layout` tensors.
7. **`gl.permute(x, (1, 0))`** works for transposing 2-D tiles; preserves layout semantics after subsequent convert_layout.
8. **`gl.dot_fma` warning `Large dot FMA instruction size 16x128x512 may have slow compile times`** is expected for these shapes. Compile takes ~10 s first call; cached thereafter.

## Why a fresh Gluon rewrite (vs incremental)?

Motivation: exp_21 exhausted the easy Triton knobs (cluster-sync requires ≥ Triton 3.8 or heavy PTX scaffolding; PDL needs compiler-emitted `griddepcontrol`; num_warps/num_stages explored; BLOCK_N explored). The remaining performance levers on Hopper/Blackwell (TMA, `tcgen05_mma`, mbarrier, warp specialization) are first-class in Gluon but not in plain Triton.

Iteration 1 (this experiment) is the correctness bring-up. Follow-on experiments will replace `dot_fma` with `bw.tcgen05_mma` and pipeline with `bw.mbarrier`, which should bring large-T back to and below exp_18.

## Learnings

- **Gluon is a viable substrate on Triton 3.7 B200.** 18 probes → one working split kernel. No CUDA, no external libraries, no web search. All API discovery done by probing.
- **`dot_fma` is ~60–80× slower than `tl.dot` for our shapes**, which is expected — it's software FMA. The Gluon win is not in dot_fma; it's in `tcgen05_mma` and the ability to explicitly control the mbarrier/pipeline/cluster structure around it.
- **Separate split + combine launches work** without the atomic barrier. The atomic barrier from exp_15/18 saves ~2 µs of launch overhead; we'll re-add it (or use `bw.mbarrier` with clusters) once dot_fma → tcgen05_mma lands.
- **`DotOperandLayout(..., k_width=0)` is mandatory with blocked parents.** Passing any other value (None, 1, 4) errors. Only learned this by systematically probing.
- **Gluon `FMA` op rejects mixed operand dtypes**, unlike `tl.dot`. Mandatory pre-cast to a common dtype (fp32) before the layout convert. bf16 in → fp32 via `.to(gl.float32)` preserves the input values exactly (bf16 is a subset of fp32).

## Next directions

1. **Replace `gl.dot_fma` with `bw.tcgen05_mma` for Q @ K^T and P @ K.** Requires allocating a tensor-memory descriptor (`bw.alloc_tmem`) and staging K in shared memory first. Expected: restore ~0.016–0.020 ms large-T latency, matching exp_18.
2. **Re-fuse split+combine using `bw.mbarrier`** once dot → tcgen05 is in place. Avoids the second kernel launch (~2 µs recovery).
3. **`num_ctas=NUM_SPLITS` cluster with `barrier.cluster.arrive/wait`** — only possible in Gluon on Triton 3.7+. Replaces the atomic-barrier with hardware cluster sync.

## Reverted? Kept? — Kept, not a new best.

This kernel is saved at `solution/triton/sparse_fused.py` and mirrored at `experiments/exp_22/sparse_fused.py`. It is **~80× slower than exp_18 on large T**, so for the leaderboard it is worse than exp_18. However, it is the foundation for Gluon-based optimization experiments (exp_23+). If a Gluon iteration can't beat exp_18 within a few experiments, revert `solution/triton/sparse_fused.py` back to exp_18's version.
