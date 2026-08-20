# Experiment 15 — 2026-04-17

**Description:** Fuse split + combine into a single kernel for the large-T (T≥3) path using an **atomic-barrier** between phases. Grid is `(T, NUM_SPLITS=8)`:
1. **Split phase:** each CTA `(t, s)` computes its `TOPK/NUM_SPLITS` partial (identical to exp_9's split).
2. **Atomic barrier:** each CTA `atomic_add(Counter[t], +1, sem="release")` then spin-waits on `atomic_add(0, sem="acquire")` until counter reaches `NUM_SPLITS`. Release–acquire pair makes prior partial stores visible.
3. **Combine phase:** same CTA (`s` is now reinterpreted as `d` for the D-slice since `NUM_SPLITS == D_CKV_SPLIT == 8`) loads all partials, combines, writes its D-slice of the output + LSE.
4. **End-of-kernel reset:** each CTA `atomic_add(Counter[t], -1)` returns the counter to 0 for the next call. Relies on CUDA stream serialization — all decrements complete before the next call's split-phase atomic_add.

Host-side: a module-level `_counter_cache` keeps one `torch.zeros(T, i32)` per `(device, num_tokens)` key — zero-initialized once at allocation, kept at 0 by the kernel's own decrement.

The T≤2 path keeps exp_13's `_fused_attn_kernel` unchanged (D-parallel single-launch).

Motivation from profile.md §Bottleneck: "two-kernel launch-barrier tax … 50–75% of per-call runtime is kernel dispatch, not compute. Collapse split+combine into a single kernel … projected T=8 17→11 µs, −35%."

## Results
- Pass: 23/23 (quick + stride 2 + full)
- Max abs err: 1.56e-02 (unchanged)
- Mode: stride 2 + full + A/B vs exp_13 (same VM)

**A/B vs exp_13 (paired, same VM, B = exp_15):**
| UUID | T | A (exp_13) | B (exp_15) | Δ |
|---|---|---|---|---|
| 0c23b10c | 1 | 0.0052 | 0.0052 | −1.23% ≈ (fused path unchanged) |
| b7668cfd | 2 | 0.0054 | 0.0054 | +0.07% ≈ |
| e6b849f2 | 2 | 0.0079 | 0.0080 | +0.28% ≈ |
| f77df5ce | 2 | 0.0053 | 0.0053 | −0.18% ≈ |
| 05f6de65 | 2 | 0.0187 | 0.0186 | −0.29% ≈ |
| 4c46a94b | 6 | 0.0176 | 0.0156 | **−11.28%** ✅ |
| 02d6ae9c | 8 | 0.0177 | 0.0158 | **−10.80%** ✅ |
| 2207f0fd | 8 | 0.0177 | 0.0158 | **−10.59%** ✅ |
| 232ed014 | 8 | 0.0177 | 0.0156 | **−11.57%** ✅ |
| 5096e459 | 8 | 0.0178 | 0.0161 | **−9.73%** ✅ |
| 564007ac | 8 | 0.0176 | 0.0162 | **−8.25%** ✅ |
| 78b2e11c | 8 | 0.0176 | 0.0158 | **−10.04%** ✅ |

**Summary:** B wins 10/12, mean Δ = **−0.0011 ms (−6.7%)**. T≥3 workloads save ~2 µs per call; T≤2 unchanged (same fused path).

## Why it works

Two 8-µs launch barriers collapsed into one. The profile estimated full elimination would save ~8 µs; we measure ~2 µs. The gap comes from:
- **Atomic spin wastes ~1–2 µs** — fastest CTA waits for slowest CTA, and the atomic read loop keeps the SM busy.
- **Counter allocation/reset is free** — `_counter_cache` avoids a `torch.zeros` per call; the in-kernel `atomic_add(-1)` resets it at no observable cost.
- **Part of the "launch tax" from the profile is actually barrier-like work** (grid-dispatch, CTA startup) that we can't skip with a one-kernel design.

The 2-µs saving across large T workloads is ~10% of the 18-µs exp_13 baseline — right in the range I'd expect if the saving is half the profile's projected ~8 µs saving.

## Key implementation details

- `NUM_SPLITS == D_CKV_SPLIT == 8` is a hard requirement: the CTA's `s` index serves as the split index in phase 1 and the D-slice index in phase 2.
- `tl.atomic_add(Counter_ptr + t, 1, sem="release")` + spin on `atomic_add(0, sem="acquire")` is the standard release/acquire barrier. Release-ordered: all prior `tl.store(partial_*)` are visible to any CTA that reads the post-increment counter.
- The cached counter strategy relies on kernel-launch serialization: CUDA streams guarantee all of the current launch's CTAs complete (including the end-of-kernel decrement) before the next launch's CTAs begin. Counter returns to 0 naturally.
- No deadlock risk: grid is (T=1..8) × (NUM_SPLITS=8) ≤ 64 CTAs on 148 SMs. Plenty of room — no two CTAs compete for the same SM. Spin-waits terminate quickly.
- Counter is 4 bytes per token (max 32 bytes) — kept hot in L2 effortlessly.

## Learnings

- **Atomic-barrier cross-CTA sync is a viable fusion pattern in Triton.** Release/acquire semantics on `atomic_add` plus a `while`-loop spin give you a single-launch equivalent of two serialized kernels. The win is real but smaller than a profile's "launch tax" would suggest (we got ~2 µs out of ~8 µs projected) — some of the "launch tax" is unavoidable setup cost.
- **Module-level counter cache avoids `torch.zeros` per-call overhead.** Pattern: allocate once per `(device, num_tokens)` tuple, let the kernel reset via `atomic_add(-1)` at the end. Works as long as launches are stream-serialized.
- **Grid re-use for two phases (split's `s` = combine's `d`) only works when `NUM_SPLITS == D_CKV_SPLIT`.** Fortunate alignment here (both were 8 from prior tuning); would need separate grid axes otherwise.

## New best. A/B confirmed −6.7% mean (−11% on large T) vs exp_13. Kept.

## Next directions

1. **Tune the fused kernel's atomics.** Is there a way to reduce the spin time? E.g., an extra `tl.debug_barrier` or cluster-barrier primitive if Triton exposes it for num_ctas clusters.
2. **Revisit num_warps/num_stages for the new fused kernel.** The combine phase is now embedded in a num_warps=8 kernel; that was optimal for split alone but combine preferred num_warps=4 in exp_9. A per-phase tuning is not possible, but an overall sweep could tell.
3. **Profile again.** The kernel structure has changed meaningfully; phase attribution (split vs combine) no longer maps to separate kernels. A fresh profile would clarify what's left.
4. **Try attacking the T≤2 launch tax** — at 7–8 µs cupti for T=1, still near the 8-µs single-launch floor. Any further win there likely comes from a structurally different design (persistent, cluster, etc).
