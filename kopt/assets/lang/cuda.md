## Language: CUDA

- **Stay in CUDA C++.** No switching to Triton or another DSL mid-run.
- **External kernels are off-limits.** Don't call into `flashinfer`, `deep_gemm`, `cuBLAS`, or
  `CUTLASS`-provided kernels as the solution — the kernel must be yours. Reading them for
  technique is fine.

### Progression

Correct naive kernel → coalesced global loads → shared-memory tiling → register blocking →
async copy / pipelining → warp-level primitives (MMA, shuffles). Don't skip structural wins for
micro-tuning.

### Tuning knobs

Block/grid shape, threads per block, shared-memory budget, registers per thread (occupancy),
`__launch_bounds__`, unroll factors, vectorized access width (`float4`), async-copy stages.
Measure occupancy rather than predicting it.

### Numerical hazards

- Accumulate in f32 (or higher) even when inputs and outputs are bf16/fp16; low-precision
  accumulation drifts over long reduction chains.
- Tensor-core paths (`wmma`/`mma`) have their own precision rules — verify against the reference
  before tuning them.
- `-use_fast_math` changes results. Don't enable it to win a benchmark.

### Escape hatch

After 15-20 iterations with no improvement, if you are certain you are stuck, spin up a
sub-agent with fresh context to restructure the kernel around a different decomposition, then
re-add optimizations one at a time based on prior experiments. Try 5-10 more iterations before
finalizing.
