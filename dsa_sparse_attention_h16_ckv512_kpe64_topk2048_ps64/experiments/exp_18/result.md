# Experiment 18 — 2026-04-17

**Description:** Per plan.md (research agent). Replace the spin-poll's RMW `tl.atomic_add(Counter_ptr + t, 0, sem="acquire")` with a pure `tl.load(Counter_ptr + t, volatile=True)`, followed by a `tl.debug_barrier()` before combine. Producer's `atomic_add(+1, sem="release")` and end-of-kernel `atomic_add(-1)` unchanged. Motivation from `profile.md`: large-T spin = 2.23 µs, 14% of total; swapping the poll from RMW → volatile load removes L2 atomic-unit serialization on the hot counter line.

## Results
- Pass: 23/23 (quick + full)
- Max abs err: 1.56e-02 (unchanged)
- Mode: quick + full + A/B vs exp_15 (two independent paired-VM runs)

**A/B vs exp_15 (paired, same VM, B = exp_18):**

Run 1: B wins **9/12**, mean Δ = **−0.0001 ms (−0.6%)**
Run 2: B wins **10/12**, mean Δ = **−0.0001 ms (−0.6%)** (stability re-run)

Large-T (T≥6) wins consistent across both runs:
| UUID | T | Run-1 Δ | Run-2 Δ |
|---|---|---|---|
| 02d6ae9c | 8 | −1.03% | −0.61% |
| 2207f0fd | 8 | −0.10% | −1.29% |
| 232ed014 | 8 | −1.21% | −1.09% |
| 4c46a94b | 6 | −0.85% | −1.02% |
| 5096e459 | 8 | −0.88% | −0.74% |
| 564007ac | 8 | −0.91% | −0.44% |
| 78b2e11c | 8 | −1.40% | −1.28% |

Small-T (T≤2) are all ties or very-small noise (±0.2 µs range). Fused T≤2 path is unchanged so that's expected.

## Why it works

`tl.atomic_add(Counter_ptr, 0, sem="acquire")` issues PTX `atom.add.acquire` with
value=0 — still an L2 atomic-unit RMW. When 8 CTAs hammer the same counter line in a
polling loop, the atomic unit serialises readers. `tl.load(..., volatile=True)` emits
`ld.volatile` (L1-bypass, L2-coherent) — a pure read, no RMW contention. The producer's
`atomic_add(+1, sem="release")` is still what establishes ordering: release on a B200
pushes prior stores to L2 before the increment becomes observable, and L2 is the
coherence point for cross-CTA traffic, so volatile loads do see the release.

`tl.debug_barrier()` post-spin prevents the compiler from speculatively hoisting combine's
`tl.load(pm_ptr_s)` above the counter wait — without it, the compiler could reorder reads
since `tl.load(volatile)` is not a PTX acquire fence. HW-level ordering is still correct.

The win is smaller than projected (plan said −0.5 to −1.0 µs; we see −0.1 to −0.2 µs on large T):
- Profile attributed 2.23 µs to "barrier spin" as an aggregate. Most of that is the slowest-CTA-wins wait duration (CTAs arriving at different times), not per-iter poll cost. Cheapening the poll only cuts the per-iter poll cost, not the wait floor.
- Still, −0.1 to −0.2 µs is ~1% on a 16-µs baseline and consistent across two independent VMs.

## Learnings

- **Volatile loads beat zero-RMW atomic reads for spin-wait polling.** When multiple CTAs poll the same counter line, `atom.add.* val=0` still serialises through the L2 atomic unit. Switching to `ld.volatile` turns the poll into pure L2 reads (no RMW contention). Correctness preserved because the producer's release fence is the only ordering point; volatile is coherent with L2 on B200.
- **Projected barrier-spin savings from profile overstate recoverable poll-cost.** Profile measured total spin = 2.23 µs; most is wait-floor (slowest CTA), not poll cost. Cheapening the poll recovers ~10% of the spin bucket (~0.2 µs), not 50%. Further wait-floor reduction needs cluster-sync or better CTA-balance.
- **`tl.debug_barrier()` is still the right companion** after a volatile-load spin. Without it, the compiler could hoist combine's loads above the wait. HW-level fence is handled by the producer's release-atomic; compiler reordering is the remaining hazard.

## New best. A/B confirmed −0.6% mean (−0.44% to −1.40% on large T) vs exp_15. Kept.

## Next directions

1. **Cluster-sync via `num_ctas=NUM_SPLITS` + inline-PTX `barrier.cluster.arrive/wait`.** Now the only remaining large-T lever. Plan says exp_19 should attempt this if exp_18 lands (it did). Projected −1.0 to −1.5 µs of the remaining ~2 µs spin, bringing large T to ~15 µs from 16 µs.
2. **Small-T Q-load overlap.** T≤2 fused_attn_kernel is at 10.15 µs with 1.92 µs Q-load. If we can overlap Q load with indices scan, ~0.3-0.5 µs possible.
3. **Persistent kernel (ruled out):** CLAUDE.md permits it technically but per-call shape doesn't benefit since we're single-call.
