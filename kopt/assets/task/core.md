# Task optimization project

Kbench owns the experiment lifecycle. The project-local harness supplies only the
repository-specific validation and measurement adapters.

## Non-negotiable rules

- Edit only the target clone and this project's `harness/`, `.omp/`, and `experiments/`.
  Never modify the auto-gpu-kernel source checkout.
- Benchmark through kbench. It runs validation before measurement, parses the standard
  results, records provenance, and uses the same current harness for both sides of A/B.
- Use absolute metric values. Confirm small changes with paired A/B on the same machine.
- Make one attributable change per iteration. A harness change is an experiment too;
  avoid mixing it with a target-source optimization.
- Harness changes may fix measurement, improve repeatability, or strengthen coverage.
  Never weaken validation to make a candidate pass or special-case benchmark inputs.
- Every result carries a harness revision. Do not compare numbers from different
  revisions without rerunning the candidates with the same current harness.
- Log every experiment via `/skill:log-experiment`, including failures.
- After logging one experiment, end the turn. The supervisor starts the next iteration.
- Do not ask the user questions. Work from the local repository and its documentation.

## Standard commands

```bash
kbench bench --quick 2>&1 | tee bench.log   # cheap validation + measurement
kbench bench 2>&1 | tee bench.log           # full validation + metric of record
kbench ab --a <git-ref>                     # same current harness, A then B
```

Kbench calls:

- `harness/validate.py --repo ... --mode quick|full --output ...`
- `harness/benchmark.py --repo ... --mode quick|full --output ...`

The scripts may call any repo-native tools they need. Kbench owns their invocation,
timeouts, output contract, result history, and A/B comparison.

## Project boundary

- `config.toml` — the human-authored task brief; do not change it during optimization.
- `<workdir>/` — the independently cloned target repository.
- `harness/` — project-local validation, benchmark adapters, fixtures, and documentation.
- `.omp/` — project-local agent instructions and skills.
- `experiments/` — plans, patches, logs, and result summaries.

The target clone and outer project are separate git repositories. Commit target-source
changes in the target repo. Commit harness, instruction, and experiment changes in the
outer project. Never force-push.
