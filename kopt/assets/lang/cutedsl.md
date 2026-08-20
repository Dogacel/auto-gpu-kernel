## Language: CuTeDSL

CUTLASS's Python DSL. Python-embedded and JIT-compiled, so it packs like Triton, but it exposes
CUTLASS's layout algebra and tiled MMA/copy atoms — far more control, and far more surface area
to get wrong.

- **Stay in CuTeDSL.** No switching to Triton or raw CUDA mid-run.
- **External kernels are off-limits.** Don't call a CUTLASS-provided kernel as the solution.
  Reading CUTLASS for technique is fine; shipping it as your answer is not.

### Read this before your first change

Prior runs of this loop measured CuTeDSL as **slower to converge than Triton**, and the cause was
process, not capability. Each optimization took several rounds to compile, then several more to
regain numerical correctness, before any latency signal existed. Three failure modes followed,
and the rules below exist to counter them:

1. **Plan amnesia** — absorbing compile-error churn crowds out the original goal.
2. **Abandoned restructurings** — verbosity makes big changes expensive, so they get dropped halfway.
3. **False infeasibility** — an abandoned attempt gets recorded as "X doesn't work" when X was
   merely tedious. That belief then blocks the idea forever.

### Rules that counter them

- **Re-read `exp_N/plan.md` after every compile fix.** State the original goal in one sentence
  before the next edit. If you cannot, you have drifted — stop and re-read.
- **Compile before you benchmark.** Get a clean build and a correct result on `--quick` first.
  Compile-fix rounds are not optimization iterations and must not be logged as experiments.
- **Keep a last-good kernel.** Before a restructuring, snapshot the working version. Revert to it
  rather than pressing on through a fourth compile round.
- **Small diffs beat rewrites.** Change one tile, one layout, one atom at a time — the debug loop
  is long enough that coupled changes are unaffordable here.
- **When you abandon, record *why* precisely.** Write "abandoned: N compile rounds, ran out of
  budget" — never "X doesn't work" — unless you measured X working and being slower. This
  distinction is the single most important thing you write to `LESSONS.md` in this language:
  a wrongly-recorded infeasibility poisons every later iteration.

### Progression

Working naive kernel → correct tiling/layout → shared-memory staging via copy atoms → tiled MMA →
pipelining / multistage → architecture-specific atoms. Establish correctness at each step; never
carry two unverified changes at once.

### Tuning knobs

Tile shapes and the layout algebra (thread-value layouts, swizzles), copy and MMA atom selection,
pipeline stage count, shared-memory budget, cluster/CTA shape on newer architectures. Verify atom
and layout APIs against the CUTLASS version actually installed in the container — the DSL surface
moves between releases, and a plausible-looking call that doesn't exist costs a full compile round.

### Numerical hazards

- Accumulate in f32 even when inputs and outputs are bf16/fp16.
- MMA atoms have fixed operand and accumulator types; a mismatch fails at compile time if you are
  lucky and silently degrades precision if you are not. Check the atom's types against the
  reference semantics.
- Layout errors read as numerical errors. When output is wrong but plausible, suspect the
  thread-value layout or a swizzle before suspecting arithmetic.

### Escape hatch

If 15-20 iterations produce no improvement, spin up a sub-agent with fresh context to rebuild the
kernel around a different decomposition, re-adding optimizations one at a time. If the loop is
still churning on compile errors rather than performance after that, say so plainly in
`LESSONS.md` — that is a real finding about the language, not about the kernel.
