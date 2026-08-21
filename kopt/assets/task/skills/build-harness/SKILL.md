---
name: build-harness
description: Generate the repository-specific validation and benchmark adapters that kbench will orchestrate. Use once when harness/prepared.json is absent.
---

# build-harness

Read the verbal brief in `config.toml` and `AGENTS.md`, inspect the untouched target
repository, and generate the two adapters kbench needs. Do not optimize the target yet.

## Boundary

The target clone is read-only during this turn. Record its HEAD and
`git status --porcelain` first. You may read source, tests, documentation, build files,
and history, and run existing commands. Do not edit tracked files, commit, switch
branches, or run tools that rewrite lockfiles. Remove untracked files you create and
finish with the same HEAD and status.

Write only under `harness/` and `.omp/`. Never access or modify auto-gpu-kernel itself.

## Understand the task

Determine from the brief and repository:

- the target code the optimizer may edit and the public behavior it must preserve;
- the cheapest meaningful correctness check and the comprehensive correctness check;
- the cheap measurement useful during iteration and the representative full metric;
- the metric unit and whether lower or higher is better;
- required setup, warmup, synchronization, seeds, caches, and noise controls.

Prefer repo-native tests and benchmarks. The adapters may wrap any local command, but
they must judge the target repository passed via `--repo`, not a copied implementation.

## Generate `harness/validate.py`

It must accept exactly these common arguments:

```text
--repo PATH
--mode quick|full
--output PATH
```

`quick` should be the cheapest reliable correctness gate. `full` should cover the
behavior promised in the user's validation description. Write a JSON object to
`--output`, even on failure:

```json
{
  "passed": true,
  "details": "short human-readable summary",
  "checks": {"optional": "stable structured details"}
}
```

Exit zero only when validation passed. Validation must not contain timing data, depend
on benchmark-specific shortcuts, or trust a success value supplied by the code being
optimized when it can check the observable result independently.

## Generate `harness/benchmark.py`

It accepts the same three arguments. `quick` may use fewer inputs or samples but must
measure the same objective as `full`. `full` is the metric of record described by the
user. Kbench runs validation first, so this script only measures. Write:

```json
{
  "value": 1.234,
  "unit": "seconds",
  "lower_is_better": true,
  "samples": [1.23, 1.24, 1.232]
}
```

`value` must be a finite number. `unit` and `lower_is_better` must be identical for
quick and full. `samples` is optional and uses the same unit as `value`. Warm up before
measuring, synchronize asynchronous work, and keep setup or compilation outside the
timed region unless the user's objective explicitly includes it.

## Document and adapt

Create `harness/README.md` describing quick/full behavior, validation, the metric,
noise controls, dependencies, and assumptions. Helper scripts and reference fixtures
belong under `harness/` as well.

Keep the existing kbench principles in `.omp/AGENTS.md` and the generic skills. Tailor
them only where concrete repository paths, commands, phases, or metric terminology make
the optimizer more effective. Do not replace the kbench-owned quick/full/A-B lifecycle
with a second mode system.

Do not create `harness/prepared.json`; kopt writes it after both pristine runs pass.

## Check

Run both scripts directly for quick mode, and full mode when practical. Inspect their
JSON and clean exploratory outputs. Confirm the target HEAD and status are unchanged.
Return a short summary; kopt will then run both modes through kbench.
