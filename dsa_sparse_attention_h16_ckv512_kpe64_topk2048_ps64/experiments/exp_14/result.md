# Experiment 14 — 2026-04-17

**Description:** num_warps=4 (down from 8) for the fused kernel. Hypothesis: with BLOCK_N_FUSED=64 the per-iter logits tile is [H=16, BLOCK_N=64] = 1024 elements; 4 warps × 32 threads = 128 threads = 8 elements/thread might fit better than 8 warps. Register pressure on acc [H=16, BLOCK_D=64] and the two kc loads (full + slice) was the main concern.

## Results
- Pass: 12/12 (quick + A/B)
- Max abs err: 1.56e-02 (unchanged)

**A/B vs exp_13 (paired, same VM, B = exp_14):**
| UUID | T | A (exp_13) | B (exp_14) | Δ |
|---|---|---|---|---|
| 0c23b10c | 1 | 0.0052 | 0.0055 | **+5.51%** ❌ |
| b7668cfd | 2 | 0.0055 | 0.0057 | +4.40% ❌ |
| f77df5ce | 2 | 0.0054 | 0.0056 | +4.01% ❌ |
| e6b849f2 | 2 | 0.0080 | 0.0083 | +3.32% ❌ |
| 05f6de65 | 2 | 0.0186 | 0.0194 | +3.90% ❌ |
| T≥3 workloads | 6–8 | — | — | ≈0 (noise; same code path) |

**Summary:** B wins 3/12 (all ties on the unchanged split-path), mean Δ = **+0.0002 ms (A faster)**. **All 5 fused-path workloads regress 3–5%.** num_warps=8 is still optimal for the fused kernel at BLOCK_N=64.

## Why it regressed

With BLOCK_N_FUSED=64 and H=16:
- Logits tile [16, 64] = 1024 bf16 → at num_warps=4 each warp owns 256 elements (2 heads × 64), at num_warps=8 each warp owns 128 (1 head × 64).
- num_warps=8 maps each warp to exactly 1 head — optimal for the softmax reduction (cross-lane sums stay within a warp for max/sum).
- num_warps=4 has warps spanning 2 heads, so each softmax reduction needs extra cross-warp coordination through shared memory.

Also: D-contraction on the output accumulator `acc [H=16, BLOCK_D=64]` — at num_warps=4, each warp holds 2×32 slab; at num_warps=8, 1×32 slab. Less register pressure per warp but more total warps/CTA. The gather load for kc [64, 512] bf16 = 64 KB prefers more concurrent issuers (more warps).

## Learnings

- **num_warps=8 is the sweet spot for the fused kernel at both BLOCK_N=128 (exp_11) and BLOCK_N=64 (exp_13).** Don't re-tune num_warps when changing BLOCK_N — the mapping of warps to H-heads drives the optimum, not BLOCK_N.
- **The "2 cache-warm num_warps" for H=16 are {8, 16}, not {4, 8}.** 8 warps × 1 head each is ideal for softmax reductions. 4 warps forces each warp to serialize 2-head reductions.

## Not a new best. Reverted to exp_13.

## Next candidates

1. **num_stages=1 on fused** (T≤2 workloads have 1–2 iters median; prefetching 2 iters ahead may waste cycles).
2. **Attack the large-T launch-barrier tax** — still the biggest lever per profile.md. Options: cluster-based fused split+combine (`num_ctas=NUM_SPLITS`), atomic-barrier between phases, or a persistent-kernel design.
3. **Revisit combine kernel tuning** — exp_9 has combine at 1.3× memcpy floor but the 2nd launch barrier contributes ~8 µs. Shrinking combine further won't help; eliminating it would.
