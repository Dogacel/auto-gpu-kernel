# /log-experiment

Log the most recent experiment. Never skip — failures are as valuable as wins.

## Pick folder

List `experiments/exp_*/`. Let `N` = highest number.
- If `exp_N/plan.md` exists without `result.md` → use `exp_N/`.
- Else → create `exp_(N+1)/`.
- No folders yet → `exp_1/`.

Never overwrite an existing `result.md`. If you'd have to, stop and ask the user.

## Write artifacts

1. Copy `solution/triton/sparse_fused.py` into the folder (same filename).
2. Copy the Modal log produced by `/benchmark` to `bench.log` in the folder.
3. Write `result.md`:

```markdown
# Experiment N — YYYY-MM-DD

**Description:** what changed, why. Reference `plan.md` when implementing one.

## Results
- Pass: X/Y
- Kernel latency (ms): small=S.SSS / large=L.LLL / overall=O.OOO (min / median / max)
- Reference latency (ms): R.RRR
- Max abs err: X.XXe-X  |  Max rel err: X.XXe-X
- Mode: quick | stride N | full  (| ab-vs-exp_K if A/B)

## Learnings
What was learned. What to try or avoid next. If durable cross-experiment insight, also append one line to `experiments/LESSONS.md`.
```

4. Append to `experiments/summary.md` (create with header row if missing):

```markdown
| Exp | Date | Description | Latency | Ref | Pass | Notes |
|---|---|---|---|---|---|---|
| N | YYYY-MM-DD | one phrase | O.OOO ms | R.RRR ms | X/Y | Δ% vs prior best, "new best" / "regression" / "ablation" |
```

Keep `Notes` terse. Detail lives in `result.md`.
