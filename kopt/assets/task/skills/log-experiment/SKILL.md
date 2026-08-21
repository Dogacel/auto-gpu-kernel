---
name: log-experiment
description: Record one source or harness experiment with candidate and harness provenance, then commit it in the correct repository.
---

# log-experiment

Log the most recent attempt, including failures.

## Pick the folder

Use the highest `experiments/exp_N/` that has `plan.md` but no `result.md`; otherwise
create the next numbered folder. Never overwrite an existing `result.md`.

## Preserve the evidence

1. Save the uncommitted target diff as `change.patch`. If this was a harness experiment,
   also save `git diff -- harness .omp` as `harness.patch`.
2. Copy `bench.log` into the experiment folder.
3. Write `result.md` with:
   - description and hypothesis;
   - candidate revision and harness revision from kbench;
   - quick or full mode;
   - validation status;
   - absolute metric value and unit;
   - sample distribution when present;
   - A/B baseline, delta, and whether lower or higher is better;
   - what was learned and what to try next.
4. Append one terse row to `experiments/summary.md` with the candidate and harness
   revisions. Add durable findings to `experiments/LESSONS.md`.

## Commit in the right place

- Target-source experiment: commit changed target files in `<workdir>/` with an
  `exp_N:` message and push the configured work branch. Never force-push.
- Harness or instruction experiment: commit `harness/` and `.omp/` in the outer project.
- Commit `experiments/` in the outer project for every attempt.

Never modify or stage `config.toml` or auto-gpu-kernel. A reverted failure with no
remaining target change needs no target commit, but its experiment record is still
committed.
