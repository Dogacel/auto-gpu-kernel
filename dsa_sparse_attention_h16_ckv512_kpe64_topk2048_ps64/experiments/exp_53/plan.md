# Exp 53 plan — 2026-04-17

**Title:** Replace serial combine loop with `tl.reduce` + softmax combine_fn (tree reduction).

**Baseline:** exp_51 (NUM_SPLITS=16, static_range combine loop).

## Hypothesis

Combine's 5.12 µs bottleneck (21% of T=8 budget per profile.md) is dominated by the
**serial softmax-rescale critical path** spanning NUM_SPLITS=16 iterations. exp_52
confirmed: `static_range` unrolling gives full ILP within one chain but cannot
parallelize across iterations (each iter's (m, l, acc) feeds the next). Chain length
= 16 operations × ~30 ns/op ≈ 480 ns plus loads.

A **tree reduction** (log2(16) = 4 stages) collapses the critical path to 4 ops.
Even if the compiler serializes the reduce internally, tree-structured combines
expose per-pair independence the compiler can actually parallelize.

## Mechanism

Flash-attention softmax-combine IS associative:
```
combine((m1, l1, a1), (m2, l2, a2)) =
  (max(m1, m2), l1·α + l2·β, a1·α + a2·β)
  where α = exp2(m1 - m_new), β = exp2(m2 - m_new), m_new = max(m1, m2)
```

Associativity ⇒ `tl.reduce` tree-pattern is mathematically valid. Triton's
`tl.reduce(tuple, axis, combine_fn)` tree-reduces via the combine_fn across the
named axis.

## Implementation

1. Define module-level `@triton.jit _flash_combine_fn(m1, l1, a1, m2, l2, a2)`
   returning `(m_new, l_new, a_new)`.
2. In combine phase of `_fused_split_combine_kernel`:
   - Replace lines 148-169 (`for si in tl.static_range(NUM_SPLITS):` loop).
   - Load all partials as 3D tiles: `pm_all [NS, H]`, `pl_all [NS, H]`,
     `pacc_all [NS, H, BLOCK_D]`.
   - Call `tl.reduce((pm_all, pl_all, pacc_all), axis=0, combine_fn=_flash_combine_fn)`.

## Risks

- **`tl.reduce` mixed-rank tuple support** uncertain on Triton 3.6. If it rejects
  heterogeneous ranks (m/l 2D, acc 3D), fall back to packing acc into m/l via
  vmap, OR manually unroll 4-chain × 4-step variant.
- **Register pressure**: pacc_all is [16, 16, 32] fp32 = 32 KB residency vs
  current 2 KB rolling tile. B200 regfile 512 KB/SM, 256 threads/CTA → 128 regs
  per partial_acc element. Fine.
- **L2 gather pattern**: single 32-KB gather replaces 16× 2-KB gathers. Strided
  reads along `offs_s` (stride_pacc_s = H × D_CKV) but partial_acc is already
  L2-resident.

## Success criteria

- Correctness: `/benchmark quick` pass 2/2 with abs_err ≤ 1.6e-2 (bf16 floor).
- Perf gate: ≥5% improvement on T=7/8 workloads via A/B vs exp_51, stride-2.
- Ceiling per profile.md: combine 5.12 → ~3 µs ≈ 8% end-to-end gain on T=8.

## Fallback if `tl.reduce` rejects mixed-rank

Write 4-chain manual:
```python
m_0, m_1, m_2, m_3 = ...  # 4 separate state tensors (independent chains)
for i in static_range(4):
    # Load 4 partials (one per chain) in parallel
    # Update each chain's state independently
# Final: merge 4 chains in a 3-step tree (6, 2 serial, 1 serial)
```
Critical path: 4 + 2 = 6 steps vs current 16.
