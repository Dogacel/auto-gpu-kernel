# Experiment 21 — 2026-04-17

**Description:** Add `launch_pdl=True` kwarg to both kernel launches. Programmatic Dependent Launch (PDL) is a Hopper+ feature discovered during exp_20's `CUDAOptions` probe. Hypothesis: the benchmark harness runs ~200 back-to-back launches of our kernel in a tight loop; PDL could let each launch start its prologue while the previous finishes, shaving dispatch cost.

## Results

- Pass: 12/12 A/B (correctness ok, abs_err 1.56e-02 as expected)
- Mode: A/B vs exp_18 (same-VM paired)

```
UUID           A (ms)     B (ms)    ΔB−A (ms)        %  winner
02d6ae9c       0.0156     0.0157      +0.0000   +0.18%  A
05f6de65       0.0185     0.0185      +0.0000   +0.05%  A
0c23b10c       0.0052     0.0051      -0.0001   -1.00%  B
2207f0fd       0.0157     0.0157      +0.0000   +0.10%  A
232ed014       0.0154     0.0154      -0.0000   -0.04%  B
4c46a94b       0.0153     0.0154      +0.0001   +0.38%  A
5096e459       0.0159     0.0159      +0.0000   +0.20%  A
564007ac       0.0160     0.0160      +0.0000   +0.28%  A
78b2e11c       0.0155     0.0155      +0.0000   +0.29%  A
b7668cfd       0.0054     0.0054      +0.0000   +0.18%  A
e6b849f2       0.0080     0.0079      -0.0000   -0.40%  B
f77df5ce       0.0053     0.0053      -0.0000   -0.43%  B

Paired n=12 | B wins 4/12 | mean Δ = +0.0000 ms → A (exp_18) faster
```

## Decision: **Reverted.**

## Why it didn't help

PDL requires the *prior* kernel in the stream to explicitly emit `griddepcontrol.launch_dependents` PTX at its tail, signaling the driver to start loading the next kernel's constants/dispatch. For our kernel to benefit from PDL, the PRIOR kernel in each back-to-back iteration would need to signal.

In the benchmark harness's hot loop, the prior kernel IS us (iteration N launches our kernel, iteration N+1 also launches our kernel). But our kernel does NOT emit `griddepcontrol.launch_dependents` — Triton doesn't auto-insert it even with `launch_pdl=True`. The kwarg enables the *capability* to accept early start; it doesn't make the kernel itself emit the signal for the next launch.

So without adding a `tl.inline_asm_elementwise("griddepcontrol.launch_dependents;", ...)` call near the end of our kernel, PDL has no upstream trigger. Even with that, the next launch's prologue overlap would be ~1 µs of driver setup, not the 5 µs noop kernel-dispatch floor (which is the time CUDA takes to actually START the kernel on SMs once it's queued).

## Learnings

- **`launch_pdl=True` alone is a no-op without an upstream kernel emitting `griddepcontrol.launch_dependents`.** In a single-kernel benchmark, enabling PDL on the kernel has no effect because the prior iteration doesn't signal. To get PDL benefits, you'd need to insert the `launch_dependents` PTX at the kernel's end (via `tl.inline_asm_elementwise`) so successive launches trigger each other. Even then, the savings are ~1 µs of driver setup overlap, not the full 5 µs noop floor (which is SM-dispatch time, not driver dispatch).

- **Triton scalar kwarg knobs are fully explored.** `num_warps`, `num_stages`, `num_ctas`, `launch_pdl`, `launch_cooperative_grid` have all been tried or ruled out. Further scalar sweeps have no hypothesis supporting them.

## Next directions

Per CLAUDE.md:
> After at least 15-20 iterations and many attempts without any successful improvement, if you are 100% sure you are stuck, spin up a new sub-agent with a fresh context and ask it to re-write the kernel in Gluon.

We are at exp_21, with 7 experiments since exp_15's last meaningful win (exp_16/17/19/20/21 regressed or neutral; exp_18 marginal). All remaining Triton levers identified in profile.md are ruled out or blocked:
- Cluster-sync — CTAPlanner assertion (exp_20)
- num_warps/num_stages — exhausted
- Launch overhead — not actionable in Triton without persistent kernel (violates submission shape)
- bf16 partial_acc — regressed (L2-resident)
- BLOCK_N tuning — saturated

**Exp_22 will be a Gluon rewrite.** Fresh context, start from baseline that matches exp_18's structural design (split+combine with atomic barrier), then add back features incrementally.
