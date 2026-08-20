# Experiment 16 — 2026-04-17

**Description:** Attempted to vectorize the combine phase in `_fused_split_combine_kernel`. Replaced the `tl.static_range(NUM_SPLITS)` sequential merge with a 3D load of all `NUM_SPLITS` partials at once (`m_all [NUM_SPLITS, H]`, `l_all [NUM_SPLITS, H]`, `acc_all [NUM_SPLITS, H, BLOCK_D]`) followed by a one-shot online softmax reduction via `tl.max` + `tl.sum(axis=0)`. Motivation: eliminate the m_global serial-dependency chain and let the compiler overlap 8 loads in parallel.

## Results
- Pass: 2/2 (quick)
- Mode: quick (correctness check only — regression clear on large T)
- Max abs err: 1.56e-02 (unchanged)
- **T=1 (0c23b10c): 0.005 ms** (unchanged — fused_attn_kernel path, not touched)
- **T=8 (2207f0fd): 0.019 ms** vs exp_15 baseline 0.016 ms → **+0.003 ms (+19%) REGRESSION**

## Decision: **Reverted.**

## Why it regressed

Triton's `tl.static_range(NUM_SPLITS=8)` fully unrolls the combine loop at compile
time — the compiler sees 8 explicit loads interleaved with 8 merge steps and schedules
them with good ILP + load/store overlap. Replacing that with a 3D-tensor load + 2D
reduction:

1. **3D tensor shape `[NUM_SPLITS, H, BLOCK_D] = [8, 16, 64]` takes 32 KB fp32 in registers or shmem.** Unrolled version keeps one `acc_si [H, BLOCK_D]` (4 KB) live at a time.
2. **`tl.sum(..., axis=0)` reduction requires cross-lane shuffles or shmem transfers** (8-way reduction). Sequential unrolled accumulation uses only FMA chains.
3. **The broadcast `acc_all * alpha_all[:, :, None]` materializes a 32 KB intermediate** before the sum, doubling register/shmem pressure.
4. **Combine was already compute/L2-floored at 2.32 µs in the exp_15 profile.** No headroom; any overhead from changing code shape immediately shows up as a loss.

## Learnings

- **Serial accumulation via `tl.static_range(N)` can be faster than vectorized reduction for small N.** When N is small (≤ 16), the unrolled chain lets the compiler schedule loads and FMAs optimally. A 3D-tensor load + axis-0 reduction adds shmem traffic that isn't amortized at this size.
- **A phase that's already at its compute/L2 floor cannot be sped up by restructuring at the same level.** The exp_15 profile explicitly noted combine @ 2.32 µs was *below* the 3.48 µs memcpy floor — no room to win from restructuring within the combine.
- **Corollary to LESSON-13 `tl.static_range(N)` pitfall**: LESSON-13 says `static_range(16)` loading 32-KB acc tiles regresses vs dynamic `range()`. But at N=8 loading 4-KB tiles (this kernel's combine), `static_range` is a *win* — the unrolled version is faster than vectorized reduction. Threshold: static_range is good when `N × tile_bytes ≤ ~8 KB` per iter; bad when the unrolled body blows up register pressure.

## Next directions

Combine is frozen — don't touch it. Next levers (from profile.md):
1. **Cluster-sync to replace atomic spin** (Lever A, −1.5 to −1.8 µs projected, medium risk). Triton does not expose cluster-barrier primitives natively; would need inline PTX or Gluon rewrite.
2. **num_stages/num_warps sweep** on the fused large-T kernel (Lever D, −0.3 to −0.8 µs).
3. **Investigate whether Q-load can be overlapped** with indices scan (~1.9 µs bucket).
