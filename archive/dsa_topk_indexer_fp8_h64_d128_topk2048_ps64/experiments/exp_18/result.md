---
exp: 18
date: 2026-04-17
status: reverted (mixed)
parent: exp_10
---

# Result — BLOCK_T=128, two pages per program, big 64×128×128 MMA (reverted)

## Change

Promoted `BLOCK_T` from 64 → 128 in `score_kernel`. Each program now
handles TWO adjacent pages. Implementation:

- Grid: `(B, cdiv(max_num_pages, 2))` — halved on the p axis.
- Two `tl.load` calls for two K tiles (each [64, 128] fp8) at different
  `page_id`s from block_table.
- Concatenate via `tl.join(k0, k1)` → `tl.trans(2, 0, 1)` → `tl.reshape([128, 128])`
  (the tl.cat + tl.sort uint64 pattern failed MLIR in exp 15; the
  tl.join+trans+reshape pattern is known-good).
- Single `tl.dot(q, k.T)` on [64, 128] @ [128, 128] → [64, 128] — one
  big m64n128k128 WGMMA instead of two m64n64k128.
- Scales concatenated similarly; `abs_t < seq_len` masks tail for odd
  `max_num_pages`.
- Scores buffer padded to `cdiv(max_num_pages, 2) * 128` to avoid
  out-of-bound stores on odd case.

## Results
- Pass: 2/2 quick (exact match, abs_err=rel_err=0)
- Mode: quick + ab-vs-exp_10

## Measurement

A/B vs exp 10 (paired, same VM):

```
Paired n=16 | B wins 5/16 | mean Δ = +0.0005 ms → A faster
```

Per-workload pattern (sorted by Δ):

| uuid | max_num_pages | Δ % | class |
|---|---:|---:|---|
| 30cecff1 | 1  | **-16.07%** | smallest (B=1) |
| a876010b | 89 | **-6.22%**  | largest |
| e63194e7 | — | -3.04% | large |
| 2f3b7321 | — | -2.57% | large |
| de54c4e6 | — | -2.39% | large |
| 19e7663d | — | +0.33% | large |
| f457feb2 | — | +0.86% | — |
| 7f1cd9c2 | — | +1.04% | — |
| e49574dd | — | +2.42% | — |
| bb22d09a | — | +3.48% | — |
| 9c313fc4 | — | **+5.73%** | medium |
| e515e20a | — | **+5.79%** | medium |
| df80c00b | — | **+5.98%** | medium |
| 05775386 | — | **+6.01%** | medium |
| 6caf09cf | — | **+6.19%** | medium |
| 4c7705ad | — | **+7.34%** | medium |

## Why it partially wins and partially loses

**Wins on extremes:**
- **Smallest (B=1, 1 page)**: halved grid + single-program early-return
  saves 16%. The launch-overhead reduction matters most when total
  time is dispatch-bound (~100 µs total).
- **Largest (29×89=2581 programs → 1291 programs)**: the ~1 µs/program
  launch-overhead × ~1290 saved programs theoretically adds up to
  several µs; observed -5 µs ≈ -6%.

**Regresses on medium (6 of 16 workloads +5-7%):** consistent pattern.
Medium workloads (likely B≈16-25, max_num_pages≈15-35) have moderate
program counts where launch overhead matters less. The cost of the
bigger MMA + register pressure + the `tl.join`+`tl.trans`+`tl.reshape`
K-tile construction outweighs the grid-halving savings. Likely causes:
1. Register pressure doubles: fp32 accumulator grows from [64, 64]
   (16 KB) to [64, 128] (32 KB). This may reduce occupancy.
2. The tl.trans+tl.reshape on a [64,128,2] fp8 tile may materialize
   intermediate SMEM writes, paid even when it could be a pure layout
   hint.
3. Two loads of smaller tiles may not amortize operand fetch as well
   as the theoretical m64n128k128 MMA suggests.

## Plan success criterion result

Plan said "no workload regresses more than +5%". Six workloads regress
>5%, so the strict criterion fails. Mean Δ is also wrong direction
(+0.5 µs).

## Reverted to exp 10 state

Will try **exp 19: BLOCK_T=128 + num_warps=8** as the plan's listed
fallback (section "Risks: R1 mitigation"). The hypothesis is that
num_warps=8 provides more thread-level parallelism to hide the
register-pressure cost on medium workloads, while preserving the
wins on largest/smallest.
