---
name: optimize
description: One autonomous kernel optimization iteration — plan a single change, implement, benchmark, log. Use for the main optimization loop.
---

# optimize — autonomous optimization loop

Iteratively improve the kernel named by `config.toml`. Rules in `AGENTS.md` are non-negotiable.

## Loop

IMPORTANT: Make sure the `research` agent is called every 5-10 experiments to ensure we are not going in circles.

1. **Assess.** Read the kernel, the baseline, `experiments/summary.md`, `experiments/LESSONS.md`. For directly relevant prior attempts, read `experiments/exp_N/result.md`. If the highest-numbered folder has `plan.md` but no `result.md`, implement that plan — it's reserved (see §Folder reservation).

2. **Plan one change.** Follow the progression ladder in `AGENTS.md` §Language. Don't skip structural wins for micro-tuning. Scan `summary.md` for similar past attempts; if close, articulate what's different *this* time.

   **Study workload characteristics** — the distribution of every variable axis listed in `AGENTS.md` §This kernel, plus any exploitable structure — and branch the kernel when a regime admits a cheaper path. Input-characteristic specialization can beat a one-size-fits-all kernel by orders of magnitude. Fair game as long as the win is real and not workload gaming.

   **Minimize launches and copies.** Fold prologue/epilogue ops (initialization, padding, masking, remapping) into the main kernel rather than calling them as separate launches. Avoid `.contiguous()` when you can plumb strides into the kernel instead — each copy is both a launch and a memory round-trip.

3. **Implement.** One optimization. Preserve DPS — the outputs are pre-allocated and passed in; never allocate them inside the kernel.

4. **Validate.** `kbench bench --quick` (2 workloads: smallest + largest — catches shape-assumption bugs). Fix compile/correctness before proceeding.

5. **Measure.** `kbench bench --stride 2`. Before trusting the number:
   - **Reference-latency sanity**: if the ref latency is >30% off the moving median from recent `summary.md` rows, the VM is anomalous — re-run once.
   - **Sub-5% deltas are noise on cross-VM comparison.** Confirm with `kbench ab --a experiments/exp_<prev-best>/solution_fused.py`.
   - Report latency split into small/large groups when both are present; aggregated means hide regime-specific regressions.
   - If results are looking good or you are not certain due to noise, proceed with a full `kbench bench`.

6. **Log.** `/skill:log-experiment`. Never skip, even on failures or ablations.

7. **Decide.**
   - **Clear win** (≥5% on stride 2, or A/B-confirmed): keep, continue.
   - **Marginal**: A/B confirm or revert.
   - **Regression**: revert, try different axis.
   - **Plateau/stuck**: see §Research-agent triggers.

8. **Budget.** `--stride 2` per iteration is the default (~2-3 min). Full runs only when you have a confirmed new best and want the real number, or every ~5 iterations as a drift check.

## When stuck — investigate before guessing

Write a targeted ablation: isolate one component — one phase of the kernel, one precision setting from `AGENTS.md` §Numerical hazards, one loop unrolled or not. Log it as its own experiment. If profiling data would resolve the question, add CUDA-event instrumentation gated by an env var (e.g. `PROFILE=1`, passed with `kbench bench --env PROFILE=1`) and log the timing breakdown — these entries are some of the most valuable later.

## Measurement agents

Three specialist agents are available via the `task` tool. They communicate with you via on-disk artifacts, never via nested context — the artifact is the contract.

- **`profiler`** (reactive, on-judgment). Call when the *next* optimization depends on knowing which phase is the bottleneck — e.g. before committing to a top-K rewrite, confirm top-K actually dominates. Writes `experiments/profile.md`. Output rots after structural changes; re-run then. Don't call on a fixed cadence — call when you genuinely don't know where the time goes.

- **`workload-inspector`** (on-hunch). Call when you suspect a data-shape or distribution angle — "are most workloads batch_size=1?", "is `block_table` contiguous?", "what fraction of `(b, h)` have zero weight?". Writes `experiments/workload_profile.md`. Output is durable (trace set is static), so typically one call is enough. `research` also calls this when needed, so you don't have to pre-stage it.

- **`research`** — see triggers below. Reads both artifacts above and synthesizes the next plan. Last-resort, not routine cadence.

## Research-agent triggers

Launch via the `task` tool with agent `research` (clean context — do NOT summarize your attempts in the prompt; the agent reads from disk). Fire when **any**:

- **True plateau**: 5+ experiments within 5% of each other.
- **Correctness wall**: 3 consecutive correctness failures on different approaches.
- **About to repeat failure**: `summary.md` shows this attempt already died.
- **Out of ideas on the current axis** and not just mid-tune.

**Not** a trigger: three honest micro-wins in a row on the same axis (tile tuning legitimately does this).

The agent writes `experiments/exp_(N+1)/plan.md`. Implement it next iteration; `/skill:log-experiment` fills `result.md` into the same folder.

## When truly stuck

`AGENTS.md` §Language ends with an escape hatch for this language — a fresh-context rewrite
under a different decomposition. It is a last resort, not a mid-run pivot: reach for it only
after 15-20 iterations with no improvement.

## Folder reservation

If `exp_N/plan.md` exists without `result.md`, that folder is **reserved**. All new work lands in `exp_N` until either implemented (result.md appears) or the plan is marked abandoned by adding `abandoned: <reason>` at the top of `plan.md` — then move on to `exp_(N+1)`.
