# Experiment 46 — 2026-04-17

**Description:** Per `exp_45/result.md` next direction. Applied `eviction_policy="evict_last"` to all three partial stores in `_fused_split_combine_kernel` (lines 124-126): partial_m, partial_l, partial_acc. Hypothesis: multi-consumer pattern (1 writer per (t,s) × up-to-8 combine readers per partial_m/l; 1 reader for partial_acc but time-adjacent) — keeping these lines L2-resident across the atomic barrier could reduce combine-phase L2 miss pressure. partial_m/l are 64 B each per store; partial_acc is 16 KB per CTA = 128 KB across 8 splits per token — meaningful L2 footprint for condition (b) from LESSON-48.

## Results
- Pass: 0/2 — **PTXAS compile error**
- Mode: quick (fused-path T=1 workload passes; combine-path T=8 workload errors)

**Error:**
```
ptxas /tmp/.../sparse_fused.ptx, line 2760; error : Modifier '.evict_last' cannot be combined with modifier '.cg'
(+9 more identical errors on lines 2773, 2820, 2827, 2834, 2841, 2848, 2855, 2862, 2869)
ptxas fatal : Ptx assembly aborted due to errors
```

Ten PTX store-sites emit `.evict_last` + `.cg` which PTXAS explicitly rejects.

## Verdict

**Reverted to exp_43 baseline.** Axis structurally blocked.

## Discoveries

1. **`cache_modifier=".cg"` + `eviction_policy="evict_last"` is illegal PTX for STORES.** On loads, `.cg` + `evict_first` compiles fine (exp_43 kernel uses this combo). The asymmetry: Triton's `eviction_policy` maps to the PTX `.L1::evict_last` hint, which instructs L1 eviction priority. For stores, `.cg` ("cache global in L2, bypass L1") means the line never lives in L1 — making the L1 eviction hint undefined → PTXAS rejects.

2. **Store-side L2 eviction hint is not programmable on this path.** B200 has no PTX-level knob to say "mark this L2 line as last-to-evict" on a store. The `evict_last` hint is an L1 concept. L2 replacement is managed by hardware LRU + a few ISA-level hints that apply only to loads. Stores always go through the HW-managed L2 write path.

3. **Dropping `.cg` to enable `evict_last` would be a dual change.** `.cg` was added in exp_24 as a marginal win (stores bypass L1 since combine reader is a different SM). Dropping it to add `evict_last` would conflate two axes and couldn't attribute any measured delta cleanly. Per LESSON-25 / CLAUDE.md "one optimization per iteration" — don't stack changes to work around PTX rejection.

4. **Load-side `evict_last` is more promising.** On LOADS, `.cg` + `evict_last` compiles (PTXAS constraint is store-specific). Candidates: combine-phase partial loads at lines 151, 152, 161 — though each line is loaded once per static_range iter per CTA, so `evict_last` only helps cross-CTA sharing between combine CTAs. Modest effect at best.

## Next directions

- **Exp_47: `input_precision="ieee"` on the 3 `tl.dot` calls** (lines 101, 102, 112 in split phase and line 254 in fused kernel). Genuinely untested axis — Triton defaults to `tf32x3` which uses 3 BF16 MMAs per FP32 mul; `ieee` is a different precision pathway. Expected neutral-to-slow but could be surprising. Sanity check: verify abs_err stays < 0.02 (per CLAUDE.md numerical hazards).
- **Exp_48: `evict_last` on Q_nope/Q_pe LOADS (lines 74-75).** Q_nope [16, 512] bf16 = 16 KB, Q_pe [16, 64] bf16 = 2 KB. Loaded once per split CTA, same t shared by 8 D-parallel CTAs → cross-CTA L2 sharing. Loads have no `.cg` so `evict_last` is PTX-legal. 8-way CTA fan-out means first loader marks lines as hot, 7 subsequent loaders can hit L2 cleanly.
- **Exp_49: `evict_last` on combine-phase partial loads (lines 151, 152, 161).** partial_m[t, si, :] = 64 B per load × 8 si = 512 B; read by all 8 combine CTAs (so each line is loaded 8 times total). First loader marks as hot → subsequent 7 loaders benefit. Same mechanism as exp_48 but on produced/stored data rather than input data.
- **Not retry:** any `evict_last` on stores with `.cg`. PTXAS-blocked.
