# Plan — exp 18

## Diagnosis

Seven consecutive reverts since exp 10 (11-17) clustered into two dead-end
families: (a) `.item()`-based dynamic K (exp 13/14, structural sync barrier),
(b) attempts to out-sort torch.topk via `tl.sort` (exp 15 tile-merge, 5× blow-up
on smallest workload), and (c) sub-1-µs Python micro-trims on py_setup / alloc /
as_strided (exp 5/11/12/16/17, all A/B ties). The lessons now pin down:
torch.topk at K=2048 is **a local optimum against bitonic-sort Triton variants**
(LESSONS.md "tl.sort scales badly past 2048" and "tile-merge does not escape
BLOCK_N wall"), and the py_setup/alloc phase is **dominated by torch dispatch
machinery that cannot be trimmed from Python** (LESSONS.md "py_setup isn't
trimmable"). The `score_kernel` phase has **not been fundamentally re-tiled**
since exp 2; every knob that's been touched is a hyperparameter (num_warps,
BLOCK_K on remap). On large workloads, score_kernel is 33.8 µs = 27% of event
total — the second-largest phase after torch.topk, and the only phase left
with a clean architectural lever that hasn't been tried.

## Strategy

**Targeted refactor**: change the score_kernel tile from `BLOCK_T=64` (one page
per program, 64×128×128 MMA) to `BLOCK_T=128` (two adjacent page_ids per
program, **one 64×128×128 MMA with N=128**). **This is not exp 3's looping
design** — no `tl.static_range`, no in-kernel page loop. A single matmul of
`[64, 128] @ [128, 128] → [64, 128]` per program, loading two K tiles via two
`tl.load` calls that concatenate along the token axis before the dot.

The payoff: (1) halves program count on score_kernel (large workload 2581 →
1291 programs; launch + SM-scheduling overhead drops proportionally), (2)
matches Blackwell FP8 tensor core's preferred WGMMA N=128 shape (current N=64
leaves ~50% of TC throughput on the table for this MMA tile), (3) amortizes Q
load (8 KB fp8) and weight load (256 B fp32) across twice the output tiles.

## Actions (priority ordered)

### 1. **What**: Promote `BLOCK_T` from 64 → 128, load two pages as one K tile per program.

- Grid: `(B, cdiv(max_num_pages, 2))` — **halved on the p axis**.
- Per program: `pid_p` now indexes *pairs* of pages. Compute two page_ids:
  ```python
  page_p0 = pid_p * 2
  page_p1 = page_p0 + 1
  ```
- Load two K tiles (each [64, 128] fp8) via two `tl.load`s addressed through
  block_table lookups, then concatenate along token axis → `[128, 128]` fp8.
  Use `tl.join` + `tl.reshape` (same pattern as exp 15's topk_kernel lines
  125-127 — known to compile on this Triton build), **not** `tl.cat` (known to
  fail MLIR pipeline per LESSONS.md).
- One MMA: `tl.dot(q_fp8 [64, 128], k_fp8_T [128, 128]) → scores [64, 128]
  fp32`.
- Apply maximum(0), broadcast-multiply by `w[:, None]`, sum over H axis → `[128]
  fp32`. Multiply by scale tile (128 elements loaded from two pages' scale
  regions).
- Tail handling for odd `max_num_pages`: `page_p1 = min(page_p0 + 1, max_num_pages - 1)`,
  mask second half's stores with `abs_t < seq_len` which already exists.
- Early-return: extend to `token_start >= seq_len` for the pair's `token_start = pid_p * 2 * 64` — unchanged semantics, just halves the number of
  early-returning programs proportionally (still block-uniform on `pid_b`).

**Why**: direct attack on `score_kernel` phase (26-34 µs p50). Profile table
in `profile.md` shows score_kernel grows sub-linearly with program count (+8 µs
going from 72 to 2581 programs = +3 ns/program launch overhead) — halving
programs saves ~4 µs at the launch-overhead floor. On top of that, a 64×128×128
MMA is closer to TC-sweet-spot than 64×128×64; Triton's generated WGMMA should
see 1.5-2× the dot throughput for the dot phase. Combined estimate: 5-10 µs
off score_kernel. Applied across all regimes (every workload has
`max_num_pages >= 1`).

**Impact**: 5-10 µs off score_kernel → new mean ~0.040-0.042 ms (from 0.047).
Best case (-20%). If only launch overhead is recovered (4 µs), still -8%
from current.

## Implementation notes

- **Tail odd-page handling**: the simplest correct approach is to compute
  `page_p1 = min(page_p0 + 1, max_num_pages - 1)` for the second page load.
  When `max_num_pages` is odd and `pid_p == last`, both page_p0 and page_p1
  point to the same final page; the second half's scores get overwritten with
  the first half's values, but `abs_t < seq_len` will mask the out-of-range
  second half to `-1e30` in the final `tl.where`, preserving correctness.
  **Verify**: the stores to `scores[pid_b, token_start+64..token_start+127]`
  for the duplicated tail would need to go to positions beyond `max_scored`;
  check that `scores` is allocated with `max_scored = max_num_pages * 64`, so
  odd `max_num_pages` means the duplicate tile writes *into valid allocated
  space* (no OOB) but those values are guaranteed beyond `seq_len` for every
  batch → correctly masked to `-1e30`. **But**: if `max_num_pages` is odd, the
  grid size `cdiv(max_num_pages, 2)` gives us `(max_num_pages + 1) // 2`
  programs, so the last program writes to tokens `[last_p*2*64 ..
  last_p*2*64 + 128)`. That range extends up to `max_num_pages * 64 + 64`,
  which is **64 tokens past** `max_scored`. **Fix**: pad the `scores`
  allocation by 64 tokens (negligible KB cost), or mask the second half
  stores with `abs_t < max_scored`.

- **K load addressing**: two `tl.load` calls (one per page), each reading a
  [64, 128] fp8 tile. Concatenate along axis-0 via `tl.join(k0, k1, dim=0)`
  then reshape to [128, 128]. Alternative: write the first [64,128] into the
  top half of a [128,128] register tile, the second into the bottom half,
  avoid reshape overhead. Pick whichever compiles cleanly.

- **num_warps**: keep default (4). The 64×128×128 MMA is the same `wgmma.mma`
  instruction as a 64×64×128 one (TC handles N=128 as one op), so register
  pressure shouldn't blow up. But num_warps=8 is worth an A/B follow-up *if*
  BLOCK_T=128 wins and appears register-bound (occupancy drop observable via
  `nsight`).

- **Double-check the scale apply**: with BLOCK_T=128, scale is a [128]
  fp32 tile (loaded from two pages' scale regions; stride `stride_ksp` jumps
  by `page_bytes // 4` across page boundaries). The final sum-then-scale
  (exp 10 lesson) becomes `sum_h(scores_after_relu_w) * scale[t]` with scale
  [128] — one more fp32 element-wise multiply than BLOCK_T=64. Tiny cost.

## Risks

- **R1 (medium): register pressure on the 64×128 fp32 accumulator doubles
  the scores-tile size from 16 KB to 32 KB worth of fp32 register state.**
  Mitigation: Blackwell tensor memory (TMEM) can hold the accumulator off the
  register file if Triton emits WGMMA-async with TMEM-allocated C; verify by
  inspecting PTX or by an A/B compile. If register pressure does trigger
  spills, fall back to `num_warps=8` (accept the exp 11 regression on B=1
  small; those workloads are dispatch-bound anyway and already worst-case).
- **R2 (low): odd `max_num_pages` + duplicated last-page store writing one
  tile past `max_scored`.** Mitigation: pad `scores` allocation by +64
  tokens (256 extra bytes per batch; trivial). Alternative: mask the
  second-half stores at the kernel boundary.
- **R3 (low): correctness ties / off-by-one in `abs_t`.** Mitigation: run
  `--quick` first for 2/2 correctness check before A/B. Also verify on at
  least one `odd-max_num_pages` workload (pick one from workload_profile
  with `mp ∈ {3, 5, 7, 9, 83}`).
- **R4 (low-medium): non-contiguous page pairs cost two independent HBM
  loads.** 66% of workloads have at least some non-contiguity; the two
  `tl.load` calls both issue in parallel so latency is min(two load issue
  cycles), not sum. Bandwidth per program doubles regardless (16 KB fp8),
  but effective HBM throughput does too. Expected net: neutral to positive.

## Do not try

- **Looping over pages inside the kernel** (exp 3: `tl.static_range(PPP)`
  regressed large +85%). This plan is a **single big matmul**, explicitly
  different.
- **PPP > 2** (implied by exp 3 — register pressure compounds non-linearly).
  Start with 2 pages/program; consider 4 only if 2 decisively wins and
  there's headroom.
- **num_warps=8 or num_warps=16 first-pass** (exp 5/11 tied or regressed).
  Only revisit if BLOCK_T=128 compiles but shows clear occupancy signs of
  register-bound behavior.
- **Beating torch.topk via tl.sort / tile-merge** (exp 8 + exp 15: BLOCK_N>2048
  wall). Keep torch.topk as-is; this experiment does not touch it.
- **.item()-based dynamic K** (exp 13/14: structural sync barrier, +30-60 µs).
- **Scores buffer caching / as_strided micro-trims** (exp 16/17: tied).
- **Radix-select Triton topk** (would compete against CUB; higher complexity,
  worse iteration-0 numbers than this plan's BLOCK_T refactor). Save as a
  fallback if BLOCK_T=128 regresses.
- **Flat active-only grid / lookup table** (blocked by the same .item()
  barrier as exp 13 — CPU needs `seq_lens` for grid size).

## Success criterion

- **Correctness**: 128/128 exact match (`matched_ratio = 1.0`) on the full run.
- **A/B vs exp 10** (stride 8, paired same-VM via `scripts/ab_benchmark.py`):
  - B wins ≥ 11/16,
  - mean Δ ≤ -3 µs (-6% from 47 µs mean),
  - **no workload regresses more than +5%** (guard against small-workload
    dispatch-overhead hit from fewer-but-bigger programs).
- **Full run** (128 workloads): mean ≤ 0.043 ms (from 0.047 ms). Stretch:
  ≤ 0.040 ms if both launch-overhead and MMA-utilization gains compound.
- **Per-workload pattern**: expect largest absolute wins on large-program
  workloads (a876010b, 2f3b7321) where halving program count saves the most
  launch overhead. Expect neutral-to-small wins on small workloads where
  score_kernel is <10% of total time anyway.

## Expected magnitude

| Component | Current | Proposed | Savings |
|---|---:|---:|---:|
| score_kernel launch overhead (~3 ns × 2581 programs, large) | ~8 µs | ~4 µs | ~4 µs |
| MMA throughput (64×128×128 vs 64×64×128, N=128 ~1.8× of N=64) | ~12 µs | ~6-8 µs | ~4-6 µs |
| Q+w load amortization (loaded once per program instead of 2×) | ~1 µs | ~0 µs | ~1 µs |
| **Total estimated savings per call (large workload)** | | | **~9-11 µs** |

Net mean-latency projection: **47 µs → 38-41 µs** on the full-run mean if
the estimates hold at all regimes; 42-44 µs if the MMA gain is half as large
as estimated. Breakeven (no regression) is the failure threshold, not the
goal.

## Coordination notes

- **Iterate via `--quick` first** (2 workloads, ~2 min on Modal) for compile
  + correctness. Move to `--stride 8` A/B only on a green quick run.
- **Test on at least one odd-`max_num_pages` workload** (pick from
  workload_profile.md: 9410ad1e has `max_num_pages=1`, 30cecff1 has 1, etc.)
  — the tail-handling mask needs a real odd case.
- **If A/B shows mean Δ ≤ -3 µs but with one workload regressing >5%**,
  the likely culprit is small-workload dispatch overhead on the single
  remaining program (e.g., mp=1 → grid becomes (B, 1) — same as before,
  no change). Compare per-workload deltas against `max_num_pages` to
  diagnose.
- **Do NOT combine with any other change in this experiment**. One axis:
  the BLOCK_T promotion. num_warps, num_stages, remap BLOCK_K, alloc
  caching — all remain at exp 10 defaults.
- **Inspect generated PTX if time permits**: `TRITON_CACHE_DIR` the kernel
  build, check for WGMMA shape `m64n128k128` in the asm. If still
  emitting `m64n64k128` (two separate MMAs under the hood), Triton isn't
  recognizing the larger N; investigate `tl.dot` acc_dtype / precision
  hints.
- **If this experiment fails (`A/B tied or mean Δ ≥ 0`), the next
  candidate is a Triton radix-select top-K kernel** (user's untried
  direction #3 — complex but addresses the actual biggest phase,
  torch.topk at 49% of total). That's a 2-day implementation scope,
  justifying another 7-iteration budget.
