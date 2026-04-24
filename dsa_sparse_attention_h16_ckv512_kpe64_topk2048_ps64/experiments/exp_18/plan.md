# Plan — exp 18

## Diagnosis
3 regressions (exp_14/16/17) are local-minimum bounce-offs: `num_warps=8` is strict optimum for H=16 (exp_14/17), combine is compute/L2-floored at 2.32 µs (exp_16). Scalar-tuning is exhausted at exp_15's structure. Per profile.md, the largest recoverable bucket on large-T is the **atomic-barrier spin at 2.23 µs (14%)**.

## Strategy
**Pivot** — attack the spin directly. Lower-risk variant first: cheapen poll-cost-per-iter (swap RMW→load). Full cluster-sync via PTX is deferred because it couples a grid-layout change with a barrier change.

## Actions (priority ordered)

1. **What:** Replace the `atomic_add(Counter_ptr + t, 0, sem="acquire")` read-poll at `sparse_fused.py:130-132` with a plain `tl.load(Counter_ptr + t, volatile=True)` loop. Keep the producer `atomic_add(+1, sem="release")` (line 127) and end-of-kernel `atomic_add(-1)` (line 184) exactly as-is. Insert one `tl.debug_barrier()` between spin-exit and the first `tl.load(pm_ptr_s)` inside the combine loop, so the compiler can't speculate combine loads above the wait.

   ```python
   # lines 130-132
   count = tl.load(Counter_ptr + t, volatile=True)
   while count < NUM_SPLITS:
       count = tl.load(Counter_ptr + t, volatile=True)
   tl.debug_barrier()
   ```

   **Why:** `atomic_add(0)` is an RMW against the L2 atomic unit even when `value=0` — readers serialise on the counter line. A volatile load is a pure L2 read, no RMW contention; L2 is the coherence point so the producer's `release` is still observed. Cheapening the poll lets waiting CTAs burn fewer cycles per iteration and return sooner.

   **Impact:** **−0.5 to −1.0 µs on large T** (16.43 → 15.4–15.9 µs, −3 to −6%). Conservative vs profile's −1.5 to −1.8 µs for full cluster-sync: we only cut poll-cost-per-iter, not the wait-duration floor (slowest-CTA-wins still dominates). Small-T untouched.

2. **Safety:**
   - Producer `release` unchanged → all partial_* stores are ordered before counter increment.
   - `ld.volatile` (L1-bypass, L2-coherent) observes `release` since L2 is the B200 coherence point.
   - Triton `volatile=True` is not a PTX acquire fence; `tl.debug_barrier()` post-spin prevents compiler reordering of combine's partial_* loads above the wait exit. HW already orders observers once releases hit L2.
   - Revert trigger: `abs_err > 1.56e-02` (exp_15 baseline) or nondeterministic outputs across 100 repeat runs.
   - If Triton rejects `volatile=True`: fallback to `cache="cg"` load hint. If both rejected, revert and fall through to the fallback below.

3. **Workflow:** Change only the 3 wait-loop lines + 1 `tl.debug_barrier()` insertion. Run `--quick` for correctness, then `scripts/ab_benchmark.py::run --a experiments/exp_15/sparse_fused.py` for paired same-VM A/B (CLAUDE.md requires this for <5% deltas).

## Do not try
- **Full cluster-sync + `num_ctas=8` + PTX `barrier.cluster.*` in one experiment** — couples grid-layout change with barrier change. Reserve for exp_19 IF exp_18 lands.
- **num_warps ∈ {4, 16}** — exp_14 and exp_17 both regressed; H=16 pins num_warps=8.
- **3D vectorized combine** — exp_16 +19% regression; `static_range(8)` unrolled is optimal.
- **bf16 partial_acc** — exp_12 +3–10%; partial_acc is L2-resident, no HBM saving.
- **NUM_SPLITS ≠ 8** — exp_4 +33–50%; combine cost vs parallelism tradeoff pinned here.
- **`if`-gated hot loop** — exp_3 +25%; defeats `num_stages=2` pipelining.
- **Any combine restructuring** — already below memcpy floor, zero headroom.

## Coordination notes
Fast iteration: `--quick` correctness → ab_benchmark paired A/B, target <10 min.

**If exp_18 regresses or ties**: escalate to cluster-sync PTX in exp_19. Cluster size 8 ≤ B200 portable limit (8, no opt-in). Max 64 CTAs on 148 SMs → no deadlock. Snippet:
```python
tl.inline_asm_elementwise(
    "barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;",
    "=r", [], dtype=tl.int32, is_pure=False, pack=1)
```
Launch-site: add `num_ctas=NUM_SPLITS`; grid stays `(T, NUM_SPLITS)`. Triton's `num_ctas` clusters the last grid axis when it divides evenly, so `program_id` semantics persist.
