# Experiment 21 — Extend workload-specialized fast path to `max_num_pages == 2`

## Goal

After exp 20's win on `max_num_pages == 1` (8 workloads, -15 µs each),
extend the same fused score+sort+remap fast path to `max_num_pages == 2`
(5 additional workloads per `workload_profile_raw.json`). Expected
per-workload savings: ~30-40 µs (torch.topk + remap dispatch still dominant).

Total projected impact: 5 × 35 µs / 128 workloads = **~1.4 µs mean improvement**.

## Evidence

From `experiments/workload_profile_raw.json`:
- 8 workloads: mp=1 (exp 20 fast path)
- **5 workloads: mp=2** (exp 21 target)
- 4 workloads: mp=3 (not addressed — odd page count needs padding)
- 2 workloads: mp=4 (future extension if mp=2 wins cleanly)

## Approach

Add a second specialized kernel `fast_small_kernel_mp2` with
`BLOCK_T_TOTAL = 128` (two 64-token pages). Dispatch via existing
Python branch:

```python
if max_num_pages == 1:
    fast_small_kernel[(B,)](..., BLOCK_T=64, TOPK=topk)
    return
if max_num_pages == 2:
    fast_small_kernel_mp2[(B,)](..., BLOCK_T_TOTAL=128, TOPK=topk)
    return
# else: exp 10 path
```

## Key kernel design

One program per batch `b`. Grid `(B,)`:

1. Load `seq_len[b]`, `page_0 = block_table[b, 0]`, `page_1 = block_table[b, 1]` → int64.
2. Load Q [64, 128], w [64] (shared across pages).
3. Load K for both pages: `k0 [64, 128]`, `k1 [64, 128]` (fp8).
4. Concatenate K via `tl.join` + `tl.trans` + `tl.reshape` → `[128, 128]` fp8.
   - This is the same pattern exp 18 used. In exp 18 it regressed medium
     workloads due to register pressure; here we only run this path on
     5 specific small workloads (all seq_lens ≤ 128), so the register
     pressure tradeoff does not generalize.
5. Load per-page scales: `s0 [64]`, `s1 [64]` → combine via `tl.join → tl.reshape` to `[128]`.
6. Compute scores: `scores = tl.dot(q_fp8, tl.trans(k_128_128), out_dtype=fp32)` → `[64, 128]`.
7. `relu → mul w → sum_h → mul scale[t] → final [128]`.
8. Mask positions ≥ seq_len to −1e30.
9. Pack uint64 sort: `(mono_f32 << 32) | t_offs.to(uint64)` on `[128]`.
   - `tl.sort` at N=128 is far below the 2048 wall — safe.
10. Remap per sorted_idx:
    - `within_page = sorted_idx & 63` (equivalent to `% 64`)
    - `page_sel = tl.where(sorted_idx < 64, page_0, page_1)` (int64)
    - `token_idx = (page_sel * 64 + within_page).to(int32)`
11. Write output: full `[2048]` store of −1, then overwrite first `[128]`
    with `tl.where(t_offs < actual_topk, token_idx, −1)`.

## Risks

- **R1 (medium): tl.join + tl.trans + tl.reshape cost.** Exp 18 showed
  this pattern is ~1-2 µs overhead per program. With 5 target workloads
  and B ranging small (likely B ≤ 14), total cost is small. Per-workload
  savings (~35 µs) still dominates.
- **R2 (low): tl.sort on 128 uint64.** At N=128, well below the 2048
  wall. Should compile and run cleanly.
- **R3 (low): two-page scale load.** Need `s0 = load(k_scale[page_0])`
  and `s1 = load(k_scale[page_1])`, then combine. Similar pattern to
  K concat.
- **R4 (low): Python branch cost grows by one more comparison.**
  Adds ~0.05 µs to the 104 non-fast-path workloads. Negligible.

## Success criterion

- Correctness: 128/128 exact match.
- A/B vs exp 20: at least 1 mp=2 workload in stride-8 sample wins
  clearly (-20% or more). Others tied.
- Full run: 5 new workloads drop from ~45 µs to ~15-20 µs.

## Expected magnitude

Per-workload on fast-path hits (5 mp=2 workloads): **30-40 µs savings**.
Mean across 128 workloads: **~1.2-1.5 µs improvement** on top of exp 20.

## Followup if it wins

- Extend to `max_num_pages == 4` (BLOCK_T=256, 2 more workloads, same pattern).
- `max_num_pages == 3` needs pad-to-256 with masking — probably not worth
  the complexity for 4 workloads unless the savings stack.
- Consider a unified kernel with `NUM_PAGES: constexpr` parameter to
  collapse all small-mp paths into one dispatch, if the separate-kernel
  approach stays clean.
