## Language: Triton

- **Stay in Triton.** No CUDA, no language switching. Suspected Triton/Python bugs are almost always something else — investigate before blaming the compiler. Gluon still counts as Triton.
- **External kernels are off-limits.** Don't call into `flashinfer`, `deep_gemm`, or similar — the kernel must be yours.

### Progression

PyTorch → tiled Triton → fused → tile tuning → alternative tilings. Don't skip structural wins
for micro-tuning.

### Tuning knobs

`num_warps`, `num_stages`, `BLOCK_*`, autotune on/off. Do **not** try to predict register
pressure from static PTX — that's unreliable; empirical tile sweeps are the answer.

### Numerical hazards

- `tl.dot` precision: try `tf32x3` first; fall back to `ieee` if `abs_err` exceeds tolerance.
- High-precision operands may need an f32 accumulator; bf16 accumulation can blow up over many terms.
- When a correctness wall appears, walk precision down: `tf32x3` → `tf32` → `ieee`.

### Escape hatch: Gluon

After 15-20 iterations with no successful improvement, if you are certain you are stuck, spin up
a sub-agent with fresh context to re-write the kernel in Gluon, then re-add features one at a
time based on prior experiments. It is a fresh start, not guaranteed faster. The best solution
may be a mix — Gluon for a special case, Triton for the rest. Try 5-10 more iterations before
finalizing.
