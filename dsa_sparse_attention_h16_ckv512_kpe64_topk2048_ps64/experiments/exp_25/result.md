# Experiment 25 — 2026-04-17

**Description:** Close the exp_21 PDL loop: emit `griddepcontrol.launch_dependents` PTX via `tl.inline_asm_elementwise` at the epilogue of both `_fused_split_combine_kernel` and `_fused_attn_kernel`, re-enable `launch_pdl=True` on both launches. Hypothesis (per exp_25/plan.md): overlap ~1 µs driver/SM-dispatch setup of next kernel with current tail, projected −0.5 to −1.5 µs per call.

## Results
- Pass: 2/2 on quick (correctness unaffected — PDL is a launch-ordering change, cannot perturb numerics)
- Max abs err: within tolerance
- Mode: A/B paired vs exp_24

**A/B vs exp_24** (paired, same VM): 6/12 B wins, mean Δ ≈ 0 ms (noise/neutral).
No consistent direction on large-T. Projected −0.8 µs per call did not materialize.

## Design

PTX epilogue added to both kernels:
```python
tl.inline_asm_elementwise(
    "griddepcontrol.launch_dependents;",
    "=r", [], dtype=tl.int32, is_pure=False, pack=1,
)
```

And `launch_pdl=True` re-added to both launch sites.

`tl.inline_asm_elementwise` accepted empty `args=[]` — no dummy tensor needed.
PTX assembled cleanly on sm_100a (B200).

## Discoveries

1. **PDL + inline PTX compiles cleanly in Triton 3.6.** `tl.inline_asm_elementwise("griddepcontrol.launch_dependents;", "=r", [], dtype=tl.int32, is_pure=False, pack=1)` is the correct signature. The PTX assembles on sm_100a.

2. **CUPTI measures individual kernel duration, not cross-kernel overlap.** The benchmark harness (`scripts/run_modal.py`) records each kernel's start/stop via CUPTI events. PDL's only win is overlapping the *next* kernel's prologue with the current kernel's epilogue — this reduces wall time between launches, not duration of either launch individually. So PDL is fundamentally invisible to CUPTI-based measurement unless we account for it via wall-clock. This extends **LESSON-29** with a key corollary: PDL is not just neutrally-measured; it is *structurally invisible* to the current measurement loop.

3. **Exp_21's "no-op without PTX signal" diagnosis was incomplete.** Even with the PTX signal correctly emitted, the CUPTI harness does not reward PDL. Any win from PDL would manifest only if we switched to wall-clock timing across the 200-call hot loop.

## Verdict

**Reverted — neutral / invisible to benchmark.** A/B 6/12 (noise), mean Δ ≈ 0. The PDL mechanism is working (PTX signal emitted, `launch_pdl=True` active, compiles + runs correctly), but the measurement harness cannot see cross-kernel overlap. Reverting the change to exp_24 kernel.

## Next directions

- **Measurement mismatch means PDL is off the table** until we can move to wall-clock. But the benchmark design is fixed per CLAUDE.md ("No CUDA graph stuff. Cupti only measures CUDA runtime"), so PDL cannot be pursued further.
- **Structural changes.** Cache/launch/PTX tuning is saturated. Next axis: persistent kernel (one CTA per SM drains multiple tokens, amortizing launch cost across tokens); or fold partial_* into HBM-resident scratch that persists across calls; or reduce compute by specializing the combine reduction (currently replicates m/l state across 8 CTAs unnecessarily on T≥3 path).
- **Pivot candidates:** (a) persistent-CTA T≥3 kernel, (b) Gluon `bw.tcgen05_mma` on the Q@K^T path (research-archived at exp_22), (c) workload-inspector re-examine distribution to catch specialization opportunities we haven't hit.
