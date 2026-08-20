# /optimize — autonomous optimization loop

Iteratively improve `solution/triton/sparse_fused.py`. Rules in `CLAUDE.md` are non-negotiable.

## Loop

IMPORTANT: Make sure research agent is called every 5-10 experiments to ensure we are not going in circles.

1. **Assess.** Read `sparse_fused.py`, `sparse_baseline.py`, `experiments/summary.md`, `experiments/LESSONS.md`. For directly relevant prior attempts, read `experiments/exp_N/result.md`. If the highest-numbered folder has `plan.md` but no `result.md`, implement that plan — it's reserved (see §Folder reservation).

2. **Plan one change.** Early progression: PyTorch → tiled Triton → fused → tile tuning → alternative tilings. Don't skip structural wins for micro-tuning. Scan `summary.md` for similar past attempts; if close, articulate what's different *this* time.

   **Study workload characteristics** — `seq_len` distribution across the 128 workloads, `batch_size`, `max_num_pages`, any exploitable structure — and branch the kernel when a regime admits a cheaper path. Input-characteristic specialization can beat a one-size-fits-all kernel by orders of magnitude. Fair game as long as the win is real and not workload gaming.

   **Minimize launches and copies.** Fold prologue/epilogue ops (initialization, padding, masking, remapping) into the main kernel rather than calling them as separate launches. Avoid `.contiguous()` when you can plumb strides into the kernel instead — each copy is both a launch and a memory round-trip.

3. **Implement.** One optimization. Preserve DPS (don't allocate `output`/`lse` inside the kernel).

4. **Validate.** `/benchmark quick` (2 workloads: smallest + largest — catches shape-assumption bugs). Fix compile/correctness before proceeding.

5. **Measure.** `/benchmark stride 2`. Before trusting the number:
   - **Reference-latency sanity**: if the ref latency is >30% off the moving median from recent `summary.md` rows, the VM is anomalous — re-run once.
   - **Sub-5% deltas are noise on cross-VM comparison.** Confirm with `modal run scripts/ab_benchmark.py::run --a experiments/exp_<prev-best>/sparse_fused.py`.
   - Report latency split into small/large groups when both are present; aggregated means hide regime-specific regressions.
   - If results are looking good or you are not certain due to noise, proceed with `/benchmark full`.

6. **Log.** `/log-experiment`. Never skip, even on failures or ablations.

7. **Decide.**
   - **Clear win** (≥5% on stride 2, or A/B-confirmed): keep, continue.
   - **Marginal**: A/B confirm or revert.
   - **Regression**: revert, try different axis.
   - **Plateau/stuck**: see §Research-agent triggers.

8. **Budget.** `stride 2` per iteration is the default (~2-3 min). `full` only when you have a confirmed new best and want the real number, or every ~5 iterations as a drift check.

## When stuck — investigate before guessing

Write a targeted ablation: isolate one component (time scoring vs top-K selection vs page-table remap; swap `tf32x3` ↔ `ieee`; measure without one loop unrolled). Log it as its own experiment. If profiling data would resolve the question, add CUDA-event instrumentation gated by an env var (e.g., `DSA_PROFILE=1`) and log the timing breakdown — these entries are some of the most valuable later.

## Measurement agents

Three specialist agents are available via the `Agent` tool. They communicate with you via on-disk artifacts, never via nested context — the artifact is the contract.

- **`profiler`** (reactive, on-judgment). Call when the *next* optimization depends on knowing which phase is the bottleneck — e.g., before committing to a top-K rewrite, confirm top-K actually dominates. Writes `experiments/profile.md`. Output rots after structural changes; re-run then. Don't call on a fixed cadence — call when you genuinely don't know where the time goes.

- **`workload-inspector`** (on-hunch). Call when you suspect a data-shape or distribution angle — "are most workloads batch_size=1?", "is `block_table` contiguous?", "what fraction of `(b, h)` have zero weight?". Writes `experiments/workload_profile.md`. Output is durable (trace set is static), so typically one call is enough. `research` also calls this when needed, so you don't have to pre-stage it.

- **`research`** — see triggers below. Reads both artifacts above and synthesizes the next plan. Last-resort, not routine cadence.

## Research-agent triggers

Launch via `Agent` tool, `subagent_type: "research"` (clean context — do NOT summarize your attempts in the prompt; the agent reads from disk). Fire when **any**:

- **True plateau**: 5+ experiments within 5% of each other.
- **Correctness wall**: 3 consecutive correctness failures on different approaches.
- **About to repeat failure**: `summary.md` shows this attempt already died.
- **Out of ideas on the current axis** and not just mid-tune.

**Not** a trigger: three honest micro-wins in a row on the same axis (tile tuning legitimately does this).

The agent writes `experiments/exp_(N+1)/plan.md`. Implement it next iteration; `/log-experiment` fills `result.md` into the same folder.

## Gluon migration

IMPORTANT: After at least 15-20 iterations and many attempts without any succesful improvement, if you are 100% sure you are stuck, spin up a new sub-agent with a fresh context and ask it to re-write the kernel in Gluon. Later add features into that final kernel one by one based on your previous experiments. This is a fresh start, it is not guaranteed to be faster than triton. An ideal solution might land in-between two, a kernel written with Gluon and another written with Triton working together. Or maybe Gluon for a special case, but triton for the rest. You should try at least another 5 to 10 iterations with Gluon before you finalize your work.

## Folder reservation

If `exp_N/plan.md` exists without `result.md`, that folder is **reserved**. All new work lands in `exp_N` until either implemented (result.md appears) or the plan is marked abandoned by adding `abandoned: <reason>` at the top of `plan.md` — then move on to `exp_(N+1)`.
