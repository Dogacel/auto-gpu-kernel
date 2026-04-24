# Experiment 37 — 2026-04-17

**Description:** Monotonic counter to eliminate the kernel-end `atomic_add(-1)` in `_fused_split_combine_kernel`. Hypothesis: on T≥3 workloads the 8 CTAs per token each do `atomic_add(+1)` at the barrier and `atomic_add(-1)` at the end, and the second atomic is pure bookkeeping (resets counter to 0 for next call). Instead, track a Python-side generation number per `(device, num_tokens)` and pass `target_count = gen * NUM_SPLITS` as a runtime arg. Counter grows monotonically across calls; the spin checks `count < target_count`. Saves 1 atomic per CTA per call (~50-100 ns × 8 CTAs serialized at L2 atomic unit).

## Results
- Pass: 2/2 quick, 12/12 stride-2, 23/23 full, 12/12 A/B
- Mode: full + A/B vs exp_26 (stride-2)
- **New best** — clean -2.5% to -3.6% on all T≥3 workloads (7/7); T≤2 fused-path unchanged as expected.

**A/B vs exp_26 (paired same-VM):**
| UUID | T-class | A (exp_26) | B (exp_37) | Δ | % |
|---|---|---|---|---|---|
| **02d6ae9c** | **T=8** | 0.0160 | 0.0155 | −0.0005 | **−2.83%** |
| 05f6de65 | T=2 | 0.0186 | 0.0186 | −0.0001 | −0.48% |
| 0c23b10c | T=1 | 0.0052 | 0.0052 | −0.0000 | −0.85% |
| **2207f0fd** | **T=8** | 0.0159 | 0.0155 | −0.0004 | **−2.56%** |
| **232ed014** | **T=8** | 0.0155 | 0.0151 | −0.0004 | **−2.82%** |
| **4c46a94b** | **T=6** | 0.0115 | 0.0111 | −0.0004 | **−3.63%** |
| **5096e459** | **T=8** | 0.0162 | 0.0157 | −0.0005 | **−2.83%** |
| **564007ac** | **T=8** | 0.0162 | 0.0158 | −0.0004 | **−2.70%** |
| **78b2e11c** | **T=8** | 0.0158 | 0.0154 | −0.0004 | **−2.79%** |
| b7668cfd | T=1 | 0.0054 | 0.0054 | +0.0000 | +0.42% |
| e6b849f2 | T=2 | 0.0080 | 0.0079 | −0.0000 | −0.28% |
| f77df5ce | T=2 | 0.0053 | 0.0053 | +0.0000 | +0.12% |

Paired: B wins 10/12 (losses both <0.5% on T=1/T=2 noise), mean Δ = −0.0003 ms → B faster. **All 7 split+combine-path workloads improve −2.5% to −3.6%.** Mean improvement on T≥3: −2.88% (~0.44 µs saved per call).

## Design

```diff
@@ def _fused_split_combine_kernel( ...
     stride_out_t, stride_out_h,
     stride_lse_t,
+    target_count,
     TOPK: tl.constexpr,
@@ # Spin until barrier releases
-    count = tl.load(Counter_ptr + t, volatile=True)
-    while count < NUM_SPLITS:
-        count = tl.load(Counter_ptr + t, volatile=True)
+    # Monotonic counter: host tracks generation; target_count = gen *
+    # NUM_SPLITS. Counter grows monotonically across calls.
+    count = tl.load(Counter_ptr + t, volatile=True)
+    while count < target_count:
+        count = tl.load(Counter_ptr + t, volatile=True)
     tl.debug_barrier()
@@ # After combine output
     tl.store(out_ptrs, acc_comb.to(tl.bfloat16))
     if d == 0:
         lse_val = ...
-
-    # Decrement counter; once all NUM_SPLITS CTAs decrement, counter returns
-    # to 0 for the next kernel call.
-    tl.atomic_add(Counter_ptr + t, -1, sem="release")

# Host side:
+_generation_cache: dict = {}
-def _get_counter(num_tokens, device):
+def _get_counter_and_target(num_tokens, device, num_splits):
     key = (device, num_tokens)
     cached = _counter_cache.get(key)
     if cached is None:
         cached = torch.zeros(num_tokens, dtype=torch.int32, device=device)
         _counter_cache[key] = cached
-    return cached
+    gen = _generation_cache.get(key, 0) + 1
+    _generation_cache[key] = gen
+    return cached, gen * num_splits
```

## Discoveries

1. **The kernel-end atomic decrement is a meaningful latency cost.** Saving ~0.0004 ms (≈0.4 µs) per call on T≥3 confirms that the L2 atomic RMW on the hot counter line was a measurable tail serializer. Consistent across 7 workloads (T=6 and T=8), with the T=6 case showing the largest -3.63% — slightly bigger relative win because its baseline is smaller.

2. **Monotonic counters replace reset-pattern atomic-barriers cleanly.** The generic idiom "increment to N, spin, decrement to 0" can be replaced with "increment monotonically, spin to target gen × N" whenever a host-side generation is trivially trackable. No correctness hazard: increments within a kernel call all happen before any CTA exits; next call's launch can't start until all CTAs' increments have landed (stream serialization); and the spin target for call N+1 is gen × N higher than any count a previous call could have left.

3. **Int32 overflow safe for benchmark scope.** Max ~20K iterations × 8 splits × a handful of `(device, num_tokens)` keys → < 200K target counts, well under 2.1B. Overflow is only a concern for multi-day training which isn't this workload.

4. **T≤2 path correctly untouched.** The fused `_fused_attn_kernel` (D-parallel, no cross-CTA sync) doesn't use the counter — only the split+combine kernel does. T≤2 latencies in A/B are within ±0.5% noise, confirming the change is surgical to the targeted path.

5. **First non-revert since exp_26.** 9 consecutive reverts (exp_27 through exp_36) on the T≥3 path on various axes (num_stages, num_warps, cache modifiers, BLOCK_N, stride-partition alternatives, D_CKV_SPLIT_FUSED). This one works because it removes actual work — the `atomic_add(-1)` was pure bookkeeping, not load-bearing. That's qualitatively different from the regressions which all tried to trade one cost for another.

## Verdict

**Kept as new best.** Target latency improvement: ~-2.88% on the 7 fused-split-combine workloads (T≥3). T≤2 fused-path neutral. Correctness verified on full 23-workload run.

## Next directions

- **Fused split+combine now has zero redundant atomics.** Remaining latency components per profile.md: launch (4.94 µs), Q load (1.92 µs), split compute (5.51 µs), barrier spin (2.23 µs), combine (2.32 µs).
- **Profile outdated** after this structural change (~0.4 µs saved on the barrier path). Consider re-profiling before the next experiment — spin-wait duration may have shifted since the L2 atomic serialization pressure is lower now.
- **Plausible next Triton axes:**
  1. **Replace atomic_add(+1) with a store-based barrier** — each CTA stores `1` to its own slot in an `[NUM_SPLITS]`-sized "ready" vector per token; others spin on the vector with OR-reduction. Eliminates the remaining +1 atomic. Feasible: `Counter_ptr` becomes `[num_tokens, NUM_SPLITS]`, each CTA writes one int and spins reading all NUM_SPLITS entries. Target: -1-2% additional on T≥3.
  2. **T=1 specialization** — T=1 workloads (0c23b10c, b7668cfd) at 5.2 µs are at noop floor; zero Triton headroom remains. Unless we rewrite as a persistent kernel, no win possible.
  3. **Further reduction of combine phase overhead** — combine loads `partial_m/l/acc` via `.cg` stores from split phase; could try reading via `tl.load` with `.ca` (cache-all) since these are written and immediately re-read within the same CTA's context (each CTA reads from all NUM_SPLITS entries it didn't write, so not all L1-cache-able — partial benefit).
- **Next attempt: exp_38 will try store-based ready-vector barrier (option 1 above).** If successful, removes the last shared atomic_add on the hot path. If it fails or ties, the barrier path is exhausted and the Gluon pivot (exp_35/plan.md) becomes the next priority.
