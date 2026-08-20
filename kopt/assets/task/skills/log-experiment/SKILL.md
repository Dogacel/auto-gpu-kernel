---
name: log-experiment
description: Record the most recent optimization attempt — snapshot the change, write result.md, append to the summary index, commit and push. Use after every benchmark, including failures.
---

# log-experiment

Log the most recent experiment. Never skip — failures are as valuable as wins.

## Pick folder

List `experiments/exp_*/`. Let `N` = highest number.
- If `exp_N/plan.md` exists without `result.md` → use `exp_N/`.
- Else → create `exp_(N+1)/`.
- No folders yet → `exp_1/`.

Never overwrite an existing `result.md`. If you'd have to, stop and end the turn.

## Write artifacts

1. Snapshot the change: `git -C <workdir> diff HEAD > experiments/exp_N/change.patch`
   (before committing), and copy the primary edited file(s) into the folder.
2. Copy the benchmark log to `bench.log` in the folder.
3. Write `result.md`:

```markdown
# Experiment N — YYYY-MM-DD

**Description:** what changed, why. Reference `plan.md` when implementing one.
**Rev:** <workdir git rev from the kbench output>
**Runner:** local / <gpu> x<count>   (from config.toml)

## Results
- Pass: yes|no   Golden: match|MISMATCH (first diff)
- Metric: S.SSS s   (mode: quick | perf | full | ab-vs-<ref>)
- Forwards (ms): min / mean / median / max
- Baseline / prev best: S.SSS s → delta %

## Learnings
What was learned. What to try or avoid next. If durable cross-experiment insight, also
append one line to `experiments/LESSONS.md`.
```

4. Append to `experiments/summary.md`:

```markdown
| N | YYYY-MM-DD | one phrase | S.SSS s | yes/no | perf/full | Δ% vs prior best, "new best" / "regression" / "ablation" |
```

Keep `Notes` terse. Detail lives in `result.md`.

## Commit and push

5. In the **workdir** (the target repo): if the tree changed, `git add -A && git commit`
   with message `exp_N: <one phrase>, <metric>s` and **push to the work branch** from
   `config.toml` (`git push origin <branch>`). Reverted/failed experiments whose tree is
   back to the previous commit need no commit. Never force-push; never commit `.kbench/`.
6. In the **project root**: `git add experiments && git commit -m "exp_N: <one phrase>"`.
