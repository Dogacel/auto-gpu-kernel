# Experiment 38 — 2026-04-17

**Description:** Per-slot ready vector barrier. Replaced shared-counter `atomic_add(Counter_ptr + t, 1)` (8 CTAs → 1 address) with per-slot `atomic_add(Counter_ptr + t * NUM_SPLITS + s, 1)` (8 CTAs → 8 adjacent addresses in same cache line). Spin loads all 8 slots vectorized + `tl.sum` against `target_count = gen * NUM_SPLITS`. Hypothesis: eliminate cross-CTA L2 atomic-unit serialization on the shared counter (~300 ns potential saving per call on T≥3).

## Results
- Pass: 2/2 quick, 12/12 A/B ×2
- Mode: quick + A/B vs exp_37 (two paired runs for noise check)
- **Reverted** — uniform +0.4-1.2% regression across all workloads

**A/B vs exp_37 (two paired runs):**

Run 1:
| UUID | T-class | A (exp_37) | B (exp_38) | Δ | % |
|---|---|---|---|---|---|
| 02d6ae9c | T=8 | 0.0154 | 0.0154 | −0.0000 | −0.14% |
| 05f6de65 | T=2 | 0.0187 | 0.0188 | +0.0001 | +0.53% |
| 0c23b10c | T=1 | 0.0055 | 0.0055 | +0.0000 | +0.35% |
| 2207f0fd | T=8 | 0.0156 | 0.0156 | +0.0001 | +0.35% |
| 232ed014 | T=8 | 0.0151 | 0.0150 | −0.0000 | −0.23% |
| 4c46a94b | T=6 | 0.0109 | 0.0110 | +0.0001 | +1.35% |
| 5096e459 | T=8 | 0.0157 | 0.0158 | +0.0001 | +0.53% |
| 564007ac | T=8 | 0.0158 | 0.0159 | +0.0001 | +0.53% |
| 78b2e11c | T=8 | 0.0154 | 0.0155 | +0.0001 | +0.58% |
| b7668cfd | T=1 | 0.0055 | 0.0055 | +0.0000 | +0.47% |
| e6b849f2 | T=2 | 0.0081 | 0.0080 | −0.0000 | −0.32% |
| f77df5ce | T=2 | 0.0054 | 0.0054 | +0.0000 | +0.18% |

B wins 3/12, mean Δ = +0.0000 ms.

Run 2 (VM noise check):
| UUID | T-class | A (exp_37) | B (exp_38) | Δ | % |
|---|---|---|---|---|---|
| 02d6ae9c | T=8 | 0.0155 | 0.0156 | +0.0001 | +0.47% |
| 05f6de65 | T=2 | 0.0186 | 0.0187 | +0.0001 | +0.50% |
| 0c23b10c | T=1 | 0.0052 | 0.0053 | +0.0000 | +0.79% |
| 2207f0fd | T=8 | 0.0154 | 0.0155 | +0.0001 | +0.54% |
| 232ed014 | T=8 | 0.0150 | 0.0150 | +0.0001 | +0.40% |
| 4c46a94b | T=6 | 0.0110 | 0.0111 | +0.0001 | +0.63% |
| 5096e459 | T=8 | 0.0157 | 0.0159 | +0.0002 | +1.20% |
| 564007ac | T=8 | 0.0158 | 0.0160 | +0.0001 | +0.91% |
| 78b2e11c | T=8 | 0.0154 | 0.0155 | +0.0001 | +0.71% |
| b7668cfd | T=1 | 0.0054 | 0.0055 | +0.0000 | +0.88% |
| e6b849f2 | T=2 | 0.0079 | 0.0080 | +0.0000 | +0.48% |
| f77df5ce | T=2 | 0.0054 | 0.0054 | +0.0000 | +0.83% |

B wins 0/12, mean Δ = +0.0001 ms. Confirms regression direction.

## Design (reverted)

```diff
-    tl.atomic_add(Counter_ptr + t, 1, sem="release")
+    tl.atomic_add(Counter_ptr + t * NUM_SPLITS + s, 1, sem="release")

-    count = tl.load(Counter_ptr + t, volatile=True)
+    offs_rdy = tl.arange(0, NUM_SPLITS)
+    rdy_ptrs = Counter_ptr + t * NUM_SPLITS + offs_rdy
+    count = tl.sum(tl.load(rdy_ptrs, volatile=True))
     while count < target_count:
-        count = tl.load(Counter_ptr + t, volatile=True)
+        count = tl.sum(tl.load(rdy_ptrs, volatile=True))
     tl.debug_barrier()

# Host-side counter tensor:
-        cached = torch.zeros(num_tokens, dtype=torch.int32, device=device)
+        cached = torch.zeros((num_tokens, num_splits), dtype=torch.int32, device=device)
```

## Discoveries

1. **Same-cache-line atomic serialization persists even with distinct 4-byte addresses.** B200's L2 atomic unit on a single cache line appears to serialize atomics regardless of whether they hit the same 4-byte slot or different slots — so splitting into per-slot addresses didn't parallelize the 8 atomics as hypothesized. The theoretical win was predicated on parallel per-slot lane execution, which doesn't happen at the cache-line granularity.

2. **Vectorized 8-wide spin load + `tl.sum` costs more than single-scalar poll.** Each spin iteration now issues `ld.volatile.v8.b32` (one cache line fetch + 8 register writes) followed by a 3-stage log-sum reduction, vs exp_37's single `ld.volatile.b32`. Even though both hit the same cache line, the register pressure and reduction ops in the tight spin loop show up as measurable ~0.5-1% overhead.

3. **The spin-wait is tighter than expected.** To see ~0.5-1% of total kernel time from the spin loop, the loop must iterate enough that the per-iter cost difference matters. With spin ~1.8 µs (post-exp_37) and per-iter ~25 ns, that's ~70 iters per call. Each extra load + sum adds ~3-5 ns per iter × 70 iters = ~200-350 ns per call. Matches the observed regression magnitude.

4. **Falsifies profile.md lever D hypothesis of "per-slot barrier = -1 to -2%".** Profile's estimate assumed separate L2 atomic unit lanes parallelize same-line atomics — not the case on B200. Same-line atomics serialize through one line's tag state; only genuinely separate cache lines (128-byte stride) would parallelize.

5. **Would need 128-byte stride to test the "parallel line atomics" hypothesis.** Each slot at stride 32 int32 = 128 bytes apart → 8 cache lines. Pros: atomics truly parallel at L2 slice level. Cons: (a) spin loads 8 separate cache lines, 8× bandwidth; (b) 1 KB × num_tokens counter tensor vs 32 B × num_tokens today; (c) bigger cache footprint may evict other useful L2 lines. Net: plausibly worse overall — skipping this variant.

## Verdict

**Reverted to exp_37.** Kernel unchanged (exp_37 state preserved).

## Next directions

- **Per-slot ready-vector axis closed.** Same-line atomics don't parallelize; strided-slot would trade atomic parallelism for poll bandwidth; either way unlikely to win.
- **Atomic+barrier lever now fully exhausted.** The `atomic_add(+1, release)` is load-bearing: it provides both accumulation and release ordering cheaply. Removing it requires structural change (cluster sync, Gluon mbarrier).
- **10 reverts since exp_26, 1 win at exp_37.** The /optimize skill's Gluon threshold is 15-20. Another 5-9 runway before mandatory pivot.
- **Plausible remaining Triton axes:**
  1. **Combine-phase partial_m/l reads could use `.cg`** — currently default `.ca`. The partial_m/l/acc tensors are L2-resident (stored with `.cg` by producer, still L2). L1 doesn't help since reads come from all NUM_SPLITS slots — each CTA reads 7 slots it didn't produce, so L1 would be a cold miss each time. (Complements exp_24 which only did `.cg` on stores.)
  2. **Q load cache modifier** — Q_nope (16 KB) and Q_pe (2 KB) are loaded once per CTA but replicated 8× across D-split CTAs. Currently default cache. Try `.ca` explicit or nothing (already `.ca`).
  3. **Small-T T=1 kernel variant** — T=1 workloads at 5.2 µs are at noop floor. Explicit T=1 specialization where only 1 D-split CTA does everything (no cross-D split, but with Q/K loaded once — essentially a non-split kernel). Could save launch overhead of 8→1 CTAs. BUT exp_36 just showed D_CKV_SPLIT_FUSED=1 regresses +20% on T=1. So this path is also closed.
- **Next attempt: exp_39 — combine-phase `.cg` on partial_m/l loads.** One-liner. If ties or marginal, it's the last cheap lever.
