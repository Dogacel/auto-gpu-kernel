# Experiment 10 — 2026-04-17

**Description:** Fused single-launch D-parallel kernel. Grid = (T, D_CKV_SPLIT=8). Each CTA owns a BLOCK_D=64 slice of the output and scans the full TOPK=2048 with online softmax, writing its output slice directly. Eliminates the separate combine kernel, the 2 MB `partial_acc` HBM round-trip, and one of two 8-µs launch barriers. Based on exp_10 profile recommendation: launch-overhead-bound, fused path projected 16 → 10 µs (−35%).

Implementation: each CTA loads the *full* kc tile for the Q@K^T logits (softmax needs full D_ckv), then loads the *d-slice* of kc for the output contraction `acc = P @ kc[:, d_slice]`.

## Results
- Pass: 12/12
- Kernel latency (ms): small=0.007 / large=0.062 / overall=0.007 (min) / 0.033 (median) / 0.063 (max)
- Reference latency (ms): not profiled
- Max abs err: 1.56e-02
- Mode: stride 2 (12 workloads)

**Per-workload Δ vs exp_9:**
| UUID | T | exp_9 | exp_10 | Δ |
|---|---|---|---|---|
| 0c23b10c | 1 | 0.011 | 0.007 | **−36%** ✅ |
| b7668cfd | 2 | 0.013 | 0.007 | **−46%** ✅ |
| e6b849f2 | 2 | 0.013 | 0.007 | **−46%** ✅ |
| f77df5ce | 2 | 0.013 | 0.007 | **−46%** ✅ |
| 05f6de65 | 2 | 0.017 | 0.014 | −18% ✅ |
| 4c46a94b | 6 | 0.018 | 0.033 | **+83%** ❌ |
| 02d6ae9c | 8 | 0.018 | 0.063 | **+250%** ❌ |
| 78b2e11c | 8 | 0.018 | 0.062 | **+244%** ❌ |
| 564007ac | 8 | 0.018 | 0.063 | **+250%** ❌ |
| 232ed014 | 8 | 0.018 | 0.037 | **+106%** ❌ |
| 5096e459 | 8 | 0.018 | 0.063 | **+250%** ❌ |
| 2207f0fd | 8 | 0.018 | 0.062 | **+244%** ❌ |

**Summary:** Bimodal. Small T (1–2) wins 36–46% (launch-tax elimination dominates). Large T (6–8) regresses 2–3.5× (softmax + logits computation duplicated 8× across D-parallel programs, and kc loaded twice per iter — once full for logits, once sliced for output). **Not a new best — ablation / failed.**

## Why it regressed on large T

Each CTA performs the full 2048 TopK scan to compute a 1/8 slice of the output. Per CTA:
- `logits = q_nope @ kc^T` needs *full* D_ckv → 128 KB kc load per iter × 16 iters = 2 MB per CTA
- `acc = p @ kc[:, d_slice]` also needs another 16 KB kc_slice load per iter
- Across 8 D-programs per token: the Q@K^T work is replicated 8× (16 heads × 128 topk × 512 D_ckv × 16 blocks × 8 CTAs = 1 G MACs per token for logits alone vs 128 M MACs in split design)

Even with L2 coalescing the redundant K reads, the **per-CTA SMEM throughput and tensor-core work is 8×** the split design. On small T the launch-tax savings (−8 µs) dominates this ~3 µs of extra compute; on large T the compute blows up.

## Learnings
- **Fused D-parallel with full-K load per CTA is the wrong shape for large T.** Each CTA duplicates the softmax/logits compute (via the full-K load) which breaks the projected 16→10 µs. Only the launch-overhead savings generalizes; the restructuring adds its own cost that grows with T.
- **Small T (T≤2) is actually launch-dominated more than SM-starved.** The fused kernel saves ~8 µs launch barrier AND eliminates the 256 KB partial_acc HBM round-trip for tiny T, where the per-CTA full-scan work is fast enough that duplication doesn't hurt.
- **Hybrid dispatch is an obvious win.** Fused for T≤2, split+combine for T≥3 (or tune the cutoff empirically). Next iteration.
- **Append to LESSONS:** "Fusing split+combine into a D-parallel single-kernel replicates the logits Q@K^T work across D-programs. Each CTA must load the full D_ckv of K for logits (can't split D on that dot), so 8 D-programs each do the full softmax → total work = NUM_D_SPLITS × split-design work. Wins only when the per-token work is tiny (T≤2) and launch tax dominates."

## Not a new best. Reverting to exp_9 (split+combine, median 0.016 ms) and planning exp_11 as a hybrid dispatch.
