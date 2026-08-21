---
name: optimize
description: One autonomous optimization iteration using kbench's validation, quick/full, and A/B lifecycle.
---

# optimize

Improve the target repository described in `config.toml`. Follow `AGENTS.md`; kbench
owns how the generated validation and benchmark adapters are executed and compared.

## One iteration

1. Read `experiments/summary.md`, `experiments/LESSONS.md`, relevant prior results,
   `harness/README.md`, and the current target code. If the newest experiment folder has
   a `plan.md` without `result.md`, implement that reserved plan.

2. Choose one attributable change. Prefer structural wins—remove work, batch or fuse
   operations, avoid transfers and synchronization, improve parallelism—before tuning
   small implementation details. Check the summary so you do not repeat a failed idea.

3. Change either the target implementation or the harness, not both unless the task is
   specifically repairing an invalid harness. A harness change must improve fidelity,
   repeatability, or coverage; never weaken validation or special-case known inputs.

4. Run `kbench bench --quick 2>&1 | tee bench.log`. Kbench validates first and only
   benchmarks a passing candidate. Fix failures before trusting performance.

5. For a credible improvement, run `kbench bench 2>&1 | tee bench.log`. This full result
   is the metric of record. Confirm small deltas with
   `kbench ab --a <previous-best-ref>` so both candidates use the same current harness.
   Results from different harness revisions are not directly comparable; rerun them.

6. Run `/skill:log-experiment`, including for failures and harness repairs. Then end the
   turn so the supervisor can start the next iteration.

## When stuck

Use the profiler when the next change depends on locating the dominant phase. Use the
research agent after a real plateau, repeated validation failures, or when the proposed
idea resembles an earlier failure. These agents read the on-disk experiment record.

If profiling needs temporary instrumentation, revert it before the logged kbench run or
make the instrumentation itself the single explicit experiment.
