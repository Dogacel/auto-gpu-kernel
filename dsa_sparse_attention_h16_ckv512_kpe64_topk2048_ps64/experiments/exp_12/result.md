# Experiment 12 — 2026-04-17

**Description:** bf16 partial_acc. Store split's `acc` as bf16 to HBM (cast from fp32), and upcast to fp32 on load in combine. Halves the partial_acc HBM traffic: at T=8, 2 MB → 1 MB per round-trip. Profile attributed ~5 µs of combine time to memcpy floor; expected saving ~2.5 µs per phase.

## Results
- Pass: 12/12
- Kernel latency (ms): per A/B vs exp_11 (same VM)
- Max abs err: 1.56e-02 (unchanged — no precision loss at the tolerance boundary)
- Mode: stride 2 (quick) + A/B vs exp_11 (same VM)

**A/B vs exp_11 (paired, same VM, B = exp_12):**
| UUID | T | A (exp_11) | B (exp_12) | Δ |
|---|---|---|---|---|
| 0c23b10c | 1 | 0.0070 | 0.0070 | −0.00% ≈ (fused path unchanged) |
| b7668cfd | 2 | 0.0072 | 0.0072 | −0.01% ≈ |
| e6b849f2 | 2 | 0.0074 | 0.0074 | −0.43% ≈ |
| f77df5ce | 2 | 0.0071 | 0.0071 | −0.27% ≈ |
| 05f6de65 | 2 | 0.0151 | 0.0151 | −0.15% ≈ |
| 4c46a94b | 6 | 0.0195 | 0.0197 | **+0.77%** ❌ |
| 02d6ae9c | 8 | 0.0198 | 0.0194 | −2.23% ✅ |
| 2207f0fd | 8 | 0.0184 | 0.0203 | **+10.13%** ❌ |
| 232ed014 | 8 | 0.0188 | 0.0195 | **+3.95%** ❌ |
| 5096e459 | 8 | 0.0190 | 0.0192 | +0.62% ≈ |
| 564007ac | 8 | 0.0186 | 0.0195 | **+4.70%** ❌ |
| 78b2e11c | 8 | 0.0185 | 0.0198 | **+6.56%** ❌ |

**Summary:** B wins 6/12, mean Δ = **+0.0004 ms (A faster overall)**. Small T (fused path) unchanged as expected. Large T shows 3–10% regressions across most workloads. **Not a new best. Reverted.**

## Why it regressed

partial_acc at T=8 = 2 MB total. **Fits entirely in B200's 126 MB L2 cache.** Between split's write and combine's read, L2 already services the data — halving HBM byte volume doesn't help because we were already hitting L2, not HBM. Meanwhile, the bf16 store adds a cast on split's epilogue, and the bf16→fp32 upcast on combine's load adds another cast. Both casts are cheap individually but the downside is register pressure and an extra instruction per element; the net is a small regression.

**Lesson:** HBM-byte-reduction micro-opts are only valuable when the data actually spills out of L2. At our partial_acc size (≤ 2 MB), L2 eats it for free.

## Learnings
- **L2 absorbs short-lived intermediates.** partial_acc writes in split and reads in combine are back-to-back in time — B200's 126 MB L2 keeps the data hot, so the combine phase is *not* really HBM-bound at this byte volume.
- **memcpy-floor anchors from the profile can mislead.** The profile reported combine at 1.3× memcpy floor, but the memcpy measurement itself was HBM-resident. Our actual combine traffic is L2-served, and the "memcpy floor" comparison overstated the HBM-saving headroom.
- **Next direction for large T:** launch overhead is still the dominant cost (split+combine = 2× 8-µs launches). Solutions must reduce *launches*, not HBM bytes. Candidates: cluster_dim in-kernel combine, persistent kernel with atomic combine, or num_warps/num_stages fine-tuning.

## Not a new best. Reverted to exp_11 (hybrid dispatch, fp32 partial_acc).
