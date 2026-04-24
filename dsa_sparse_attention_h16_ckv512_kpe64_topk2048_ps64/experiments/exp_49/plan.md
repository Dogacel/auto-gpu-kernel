# Plan — exp 49

## Diagnosis

Since exp_37 (the last clear win at -2.88%), 11 experiments have produced only 2
marginal keeps (exp_43 `evict_first` on K loads, exp_48 `evict_last` on Q loads,
both ~0.3% on affected paths). All four L2 cache/eviction sub-axes are saturating:
exp_23/24/28/41 closed `cache_modifier` across K/store/combine/Q placements
(LESSON-47); exp_43/44/45/46/48 characterized `eviction_policy` across K
loads (kept), idx loads (regressed), Q loads (kept), stores (blocked by PTXAS
per LESSON-49). Atomic-barrier axis is closed at both 32-B and 128-B slot
strides (exp_38/42, LESSON-45). Gluon `bw.tcgen05_mma` is 6.1× slower than
`tl.dot` at H=16 due to MMA-tile padding (LESSON-46). Profile refresh (exp_41)
confirmed split_work=8.05 µs dominates the ~15.5 µs CUPTI kernel; ~1 µs of
software-accessible headroom remains.

**Root cause of plateau**: the optimizer has been iterating within the
**cache-hint micro-tuning axis**, which is saturated. The genuine untested
axis is a **structural change that directly exploits the workload's bimodal
valid-count distribution** (median per-token valid=33 in a 2048-wide TopK).

Under the current stride-partition (exp_26) `BLOCK_N=128` processes 128 TopK
positions per loop iter on the split kernel. For the median token (valid=33,
per-split valid≈4), a single iter has **~3% of the BLOCK_N doing useful work
and 97% masked padding**. Halving BLOCK_N to 64 halves per-iter compute and
per-iter K_ckv HBM traffic for the majority of tokens, at the cost of +1 loop
iter on high-valid outliers (iter count goes 2→3 for per-split valid ∈
(128, 192]).

`BLOCK_N=64 on the split path under stride-partition is genuinely untested`.
exp_9 doubled BLOCK_N to 128 pre-stride-partition; exp_13 halved to 64 on the
FUSED (T≤2) path only. The combination (stride-partition + BLOCK_N=64 on
split-combine) has never run.

## Strategy

**Targeted structural change.** One constexpr-scalar change to `BLOCK_N`
passed to `_fused_split_combine_kernel` in the host dispatch. Preserves all
other proven wins (stride-partition, monotonic counter, .cg on K, evict_first
on K, evict_last on Q, .cg on stores). Mirrors exp_13's pattern on the split
path.

## Actions (priority ordered)

### 1. **What:** Halve `BLOCK_N` from 128 → 64 on the split-combine kernel path

   In `solution/triton/sparse_fused.py` host `kernel()`:

   ```python
   # line 314
   BLOCK_N = 128   # current
   ```

   becomes

   ```python
   BLOCK_N = 64    # split path; matches BLOCK_N_FUSED=64 in exp_13
   ```

   Line 373 already passes `BLOCK_N=BLOCK_N` as `tl.constexpr` to
   `_fused_split_combine_kernel`. No kernel-code changes needed — the
   constexpr drives `offs_n`, `max_bn` rounding, K/P tile shapes, logits
   shape, P shape. All downstream shapes scale.

   **Critical invariant check**: `SPLIT_SIZE = TOPK // NUM_SPLITS = 2048 / 8
   = 256`. With BLOCK_N=64, max_bn can be 0, 64, 128, 192, or 256 → up to
   **4 loop iters per split** (vs up to 2 at BLOCK_N=128). Stride-partition's
   `offs_split = s + arange(256) * NUM_SPLITS` unchanged; per-iter idx load
   `s + (bn + offs_n) * NUM_SPLITS` now uses offs_n=arange(64) and strides
   through `idx_scan` at 64-elt chunks. Correctness preserved.

   **Shmem budget**: K_ckv tile under num_stages=2 = 64 × 512 × 2 bytes × 2
   stages = 128 KB; K_pe = 64 × 64 × 2 × 2 = 16 KB. Total 144 KB ≪ 228 KB
   B200 per-SM limit. No compile risk (vs exp_33 which OOMed at 321 KB for
   BLOCK_N=256).

   **MMA tile check**: `tl.dot` input shapes become `[16, 512] @ [512, 64]
   = [16, 64]` (Q@K^T) and `[16, 64] @ [64, 512] = [16, 512]` (P@K). Both
   valid MMA shapes on sm_100a; `wgmma.bf16.bf16.f32 m64n64k16` (or m=64
   with phantom rows for H=16) maps. No precision change (`tf32x3`/`ieee`
   lever is closed per LESSON-50).

   **Why:** The workload profile (`experiments/workload_profile.md`) shows
   per-token valid distribution p50=33, p90=1089. Under stride-partition
   each split sees `num_valid / NUM_SPLITS` valids, so per-split valids
   cluster in 4-140 range. BLOCK_N=128 processes these 128-at-a-time,
   masking out most of the tile. BLOCK_N=64 better matches the per-split
   valid footprint for the majority class of tokens. LESSON-19 established
   this pattern's validity on the fused path (-26% on low-valid); transfer
   to split path is untested but mechanism-analogous.

   Differs from exp_13 (which only touched fused): this targets the split
   path (T≥3 workloads, ~61% of the workload set by count, ~100% of the
   batched-latency dominant workloads). Transfer is inexact because:
   (a) per-split valid is smaller than per-token valid (so low-valid regime
   is MORE dominant on split path), and (b) iter count bound is tighter
   (max 4 vs exp_13's max 32 at fused).

   **Impact:** Expected net -1 to -3% on T≥3 workloads. Best case: tokens
   with per-split valid ≤ 64 keep 1 iter (80%+ of tokens by workload_profile
   distribution) and save ~50% of split-phase compute → potentially -4 to
   -8% on those. Worst case: high-valid outliers (p90 per-token valid=1089 →
   per-split ≈136) go from 2 iters → 3 iters, adding ~0.5-1 µs of loop
   overhead (minor regression, magnitude ≤ +2%).

### 2. **What:** Validate numerically and via A/B

   **Validation criteria for KEEP (listed in priority order):**
   - Correctness: 128/128 full benchmark passes; abs_err ≤ 1.56e-02 (matches
     baseline; byte-for-byte not required since tile shapes change).
   - Latency: A/B stride-2 vs exp_48 (current best), two runs for noise.
     **Keep gate**: T≥3 affected-path ≥ 9/14 B wins across both runs AND
     mean Δ ≤ 0 on T≥3 AND no single T≥3 workload regresses >2%.
   - Mechanism: at least one workload shows ≥ -2% with direction consistent
     across both A/B runs (rules out "narrow wins + wide regressions").

   **Revert gate** (any one triggers):
   - Any T≥3 workload regresses >2% consistently across both A/B runs.
   - Mean Δ on T≥3 > 0.
   - T≤2 path shows any regression (would indicate cross-path contamination;
     should be impossible since fused kernel unchanged).

### 3. **Coordination notes**

   - **Single variable change**: only `BLOCK_N` in host dispatch. No kernel
     edit, no new constexpr, no branch.
   - **Run order**: quick (2/2) → stride-2 A/B ×2 against exp_48 → full
     128/128 if gate passes.
   - **No profile re-run required**: the change is a block-size sweep on
     the known-dominant bucket. If keep, follow-up profile (exp_50+) would
     re-attribute split_work bucket.
   - **Reference for rollback**: `experiments/exp_48/sparse_fused.py`.
     Single-line revert: `BLOCK_N = 64 → BLOCK_N = 128`.

## Do not try

- **BLOCK_N=256**: exp_33 OOMed (321 KB shmem > 228 KB limit); even
  num_stages=1 fallback regressed +62% at the existing BLOCK_N=128 (exp_32).
- **BLOCK_N=32**: would push iter count to 8 per split — loop overhead at
  4× BLOCK_N=128 likely dominates any per-iter savings. LESSON-19 notes
  BLOCK_N_FUSED crossover at valid≈100-500; BLOCK_N=32 is below that curve.
- **Combined BLOCK_N + num_warps/num_stages change**: violates
  one-optimization-per-iter. num_stages=2 and num_warps=8 already strict
  optima (exp_14/17/19/32/34, LESSON-26).
- **Any further cache_modifier / eviction_policy placement**: 8 sub-axes
  tested to saturation per LESSONS 47/48/49/51. Axis fully characterized.
- **Atomic-barrier variants**: LESSON-45 closes at both 32-B and 128-B
  slot strides.
- **Gluon `bw.tcgen05_mma`**: LESSON-46 establishes it's 6.1× slower at
  H=16. Structurally uncompetitive.
- **num_ctas / cluster-sync**: LESSON-27 PlanCTAPass assertion with atomic
  barrier; structurally blocked.
- **`input_precision="ieee"` on bf16 dots**: LESSON-50 no-op.
- **Per-workload `num_valid` dispatch**: LESSON-20 impossible without GPU
  sync overhead exceeding any saving.

## Notes on ceiling vs current

The ~14.5 µs CUPTI Triton ceiling projected in exp_41's profile assumed
split_work=8.05 µs is irreducible (`tl.dot` at peak). This plan challenges
that assumption: if per-iter tile size is cut in half while compute-utility
stays the same, split_work budget drops to ~4-6 µs for low-valid tokens —
which are the majority. If BLOCK_N=64 wins here, the "Triton ceiling"
estimate needs revision downward to ~12-13 µs CUPTI on median workloads.

This is the last genuinely untested axis within the current split-combine
architecture. If it fails (plateau remains), the optimizer should pivot to
structural architecture changes (e.g., warp-specialization, dual-phase
kernel with separate tile sizes per phase) rather than continuing
parameter-sweep within the current shape.
