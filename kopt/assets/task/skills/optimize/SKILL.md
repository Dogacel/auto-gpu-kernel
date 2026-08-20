---
name: optimize
description: One autonomous optimization iteration — plan a single change, implement, benchmark, log. Use for the main optimization loop.
---

# optimize — autonomous optimization loop

Iteratively speed up the target repo named in `config.toml`. Rules in `AGENTS.md` are
non-negotiable — especially: one change per iteration, correctness = golden match, log
everything, push after logging.

## Loop

IMPORTANT: Make sure the `research` agent is called every 5-10 experiments to ensure we are
not going in circles.

1. **Assess.** Read `experiments/summary.md`, `experiments/LESSONS.md`, and the current
   state of the code you are optimizing (§This task in `AGENTS.md` says what and where).
   For directly relevant prior attempts, read `experiments/exp_N/result.md`. If the
   highest-numbered folder has `plan.md` but no `result.md`, implement that plan — it's
   reserved (see §Folder reservation).

2. **Plan one change.** Order of leverage:
   - **Structural / algorithmic** first: eliminate whole phases, overlap independent work,
     avoid recomputation, batch or fuse launches, cut host-device round trips.
   - **Parallelism across the hardware**: the machine's full GPU count is in `AGENTS.md`
     §This task. Work the correctness contract allows to be distributed (per-layer, per-
     phase, pipelined) is fair game as long as the golden artifact still matches exactly.
   - **Kernel-level** last: tile shapes, fusion, launch parameters — only once the
     structure is right.
   Scan `summary.md` for similar past attempts; if close, articulate what's different
   *this* time. Don't skip structural wins for micro-tuning.

3. **Implement.** One optimization, in the target repo working tree. Keep the external
   contract the task description pins (entry-point signatures, device placement of inputs
   and outputs, artifact schema) exactly intact.

4. **Validate.** `kbench bench --quick 2>&1 | tee bench.log` — the correctness gate.
   Fix compile/correctness before proceeding. A golden MISMATCH means your change altered
   the numerics: there is no tolerance to negotiate with; revert or fix the bit-exactness.

5. **Measure.** `kbench bench --mode perf 2>&1 | tee bench.log` for a clean timing signal.
   Before trusting the number:
   - Deltas under ~2% are noise even on a quiet local machine. Confirm with
     `kbench ab --a <ref-of-prev-best>` before claiming a small win.
   - If results look like a real improvement, run `kbench bench --mode full 2>&1 | tee bench.log`
     to lock in the number of record on the real workload.

6. **Log.** `/skill:log-experiment`. Never skip, even on failures or ablations. This also
   commits and pushes the change.

7. **Decide.**
   - **Clear win** (≥2% on perf, confirmed on full): keep, continue.
   - **Marginal**: A/B confirm or revert.
   - **Regression / MISMATCH**: revert the working tree (`git checkout -- .` in the
     workdir, or a targeted revert), try a different axis.
   - **Plateau/stuck**: see §Research-agent triggers.

8. **Budget.** quick+perf per iteration is the default. Full runs when you have a
   confirmed new best, or every ~5 iterations as a drift check.

## When stuck — investigate before guessing

Write a targeted ablation or a profiling probe: time one phase with CUDA events or
`torch.profiler`, gated behind an env var (pass with `kbench bench --quick --env PROFILE=1`),
and log the breakdown as its own experiment. Ad-hoc probe scripts are allowed for
investigation (run them directly), but never as a substitute for the logged `kbench` numbers.

## Measurement agents

Two specialist agents are available via the `task` tool. They communicate via on-disk
artifacts, never via nested context — the artifact is the contract.

- **`profiler`** (reactive, on-judgment). Call when the *next* optimization depends on
  knowing which phase dominates. Writes `experiments/profile.md`. Output rots after
  structural changes; re-run then.

- **`research`** — see triggers below. Reads the profile and synthesizes the next plan.
  Last-resort, not routine cadence.

## Research-agent triggers

Launch via the `task` tool with agent `research` (clean context — do NOT summarize your
attempts in the prompt; the agent reads from disk). Fire when **any**:

- **True plateau**: 5+ experiments within 2% of each other.
- **Correctness wall**: 3 consecutive golden MISMATCHes on different approaches.
- **About to repeat failure**: `summary.md` shows this attempt already died.
- **Out of ideas on the current axis** and not just mid-tune.

The agent writes `experiments/exp_(N+1)/plan.md`. Implement it next iteration;
`/skill:log-experiment` fills `result.md` into the same folder.

## Folder reservation

If `exp_N/plan.md` exists without `result.md`, that folder is **reserved**. All new work
lands in `exp_N` until either implemented (result.md appears) or the plan is marked
abandoned by adding `abandoned: <reason>` at the top of `plan.md` — then move on to
`exp_(N+1)`.
