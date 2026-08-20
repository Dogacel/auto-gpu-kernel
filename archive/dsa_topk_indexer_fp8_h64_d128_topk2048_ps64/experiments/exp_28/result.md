---
exp: 28
date: 2026-04-17
status: new best
parent: exp_26
---

# Experiment 28 — 2026-04-17

**Description:** Fused Triton radix-select top-K (`radix_topk_kernel`) replaces
`torch.topk` + `remap_kernel` on the slow path (`max_num_pages > 32`). One
program per batch: scoreless branch if `seq_len <= topk`; else 32-iteration
bit-by-bit radix-select on monotone-uint32 keys, then scatter-store via
prefix-sum addressing.

## Implementation

1. **`radix_topk_kernel`** (new):
   - fp32 → monotone uint32: `sign = bits >> 31; xor_mask = sign ? 0xFFFFFFFF : 0x80000000; mono = bits ^ xor_mask`.
   - OOB lanes (`offs >= max_scored`) forced to `mono = 0` (smallest, never selected).
   - Threshold build: for bit in 31..0, candidate = threshold | (1<<bit); accept if count(mono >= candidate) >= topk.
   - Post-threshold: strict mask (mono > thr), tie mask (mono == thr). Pick strict + first (topk - strict_count) ties by position prefix-sum.
   - Final mask cumsum → write position; scatter token_idx (fused block_table lookup) at that position with mask.
2. **Dispatch**: slow path `BLOCK_N = next_power_of_2(max_scored)` (4096 or 8192), grid = `(batch_size,)`. Replaces both `torch.topk` and `remap_kernel` calls.
3. **`score_kernel` unchanged** — scores tensor still produced, only the post-score path changed.

## Results

- Pass: **128/128**
- Kernel latency (ms) — full run: small=0.0020 / large=0.0407 / overall=**0.0199** mean (min 0.002 / max 0.084)
- Reference latency: ~0.000 ms (Modal reports 0.00x — baseline not instrumented this run)
- Max abs err: 0.00 | Max rel err: 0.00 (matched_ratio = 1.0000 across 5 trials)
- Mode: stride 8 + ab-vs-exp_26 + full

### A/B vs exp 26 (stride 8, paired, same VM)

- B wins 9/16, mean Δ = **-0.0151 ms** → B faster
- Seven slow-path workloads at **-59% to -65%** (from ~57 µs → ~22 µs):
  - 19e7663d -61%, 2f3b7321 -59%, 4c7705ad -65%, 7f1cd9c2 -64%, de54c4e6 -59%, e63194e7 -59%, f457feb2 -63%
- One regression: **a876010b +7%** (0.082 → 0.088 ms) — heaviest workload, mp=91, BLOCK_N=8192. The 32-iteration radix-select's reduction cost scales with BLOCK_N and is close to torch.topk on the very largest tiles.
- Fast-path workloads (mp ≤ 32): unchanged (tied within noise).

### Full 128-workload

| Group | Count | Mean (µs) |
|---|---|---|
| Small (fast path) | 69 | 2.00 |
| Large (slow path) | 59 | 40.73 |
| Overall | 128 | **19.85** |

vs exp 26: 27.6 µs → 19.85 µs → **-28% full-run mean**, **-29% slow path**.

## Learnings

1. **Radix-select works on Blackwell.** 32-iteration bit-by-bit threshold build avoids the BLOCK_N≥4096 `tl.sort` wall (exp 8/15). Each iteration is a `tl.sum` of a boolean tile — cheap tree reduction.
2. **Fusing remap into the top-K kernel is free.** Computing `token_idx` in the same program costs only the block_table load; saves the ~6 µs remap_kernel launch.
3. **Scatter-via-cumsum works.** `write_pos = cumsum(final_mask) - 1` + `tl.store(ptr + write_pos, val, mask=final_mask)` correctly compacts selected elements into contiguous output positions. Total selected = topk by construction (invariant from radix-select), so all 2048 output slots get written — no init needed.
4. **Heaviest workload (a876010b, mp=91) regressed.** Radix-select cost scales with BLOCK_N=8192. 32 × tl.sum on 8192 elements may be comparable to torch.topk there. Possible follow-up: 2-pass 11-bit histogram (plan's original suggestion) should be faster for large BLOCK_N because the total work per pass is O(N) plus a 2048-bucket cumsum — not 32 reductions.
5. **Scoreless-per-batch branch inside the kernel matters.** Unlike exp 27 (where the parallel score_kernel hid per-batch work), here the radix_topk is launched with grid = (batch_size,), so scoreless batches shave work off the critical path when the scoring batch doesn't fully utilize the wave.

## Next candidates

- **Exp 29**: tune radix kernel for large BLOCK_N. Try 16-iteration 2-bit radix (4 buckets per iteration, 8 iterations) or 2-pass 11-bit histogram. Goal: fix the a876010b +7% regression without losing the mp≤64 wins.
- Tuning knobs: `num_warps` on radix kernel, `BLOCK_N` specialization (compile variants for 4096 vs 8192), early-terminate the bit loop once `count == topk` exactly.
