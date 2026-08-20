# Kernel optimization project

Autonomous GPU kernel optimization. The kernel under optimization, its GPU backend, and the
benchmark settings all come from `config.toml` — read it first.

## Non-negotiable rules

- **Absolute latencies only.** Speedup ratios lie: reference latency swings 20-30% across VMs. It is normal to see (0.00x), only rely on absolute numbers.
- **Never compare latencies across backends.** A number from `fal` and a number from `modal` are different measurements. Every logged result records its backend; comparisons stay within one.
- **One optimization per iteration.** Coupled changes misattribute wins. For sub-5% deltas vs previous best, use `kbench ab` (paired, same VM) — cross-VM comparison from `summary.md` is noise.
- **Benchmark through `kbench`.** Never hand-roll a benchmark; the backend is configured, not chosen per-run.
- **Log every experiment** via `/skill:log-experiment`, including failures.
- **One optimization per turn, then stop.** After you have logged the experiment, end your
  turn. A supervisor re-invokes you immediately with fresh context — you are not ending the
  optimization, only this step of it. Continuing past the log couples changes together and
  destroys the attribution that `/skill:optimize` depends on.
- **No benchmark gaming.** No memoizing outputs, no iteration-counter tricks, no `--quick`-specific shortcuts. No CUDA graphs: the timer measures CUDA runtime, and caching input pointers or capturing graphs is forbidden. You may use event streams to debug runtimes.
- **No web access.** You work from the repo and the trace set only. Do not search the web or read remote URLs.
- **Don't ask anything to the user.** You are designed to work autonomously and the user won't answer your questions.

## Skills

| Skill | Purpose |
|---|---|
| `/skill:optimize` | Main loop — one optimization, benchmarked and logged |
| `/skill:log-experiment` | Snapshot kernel + write `result.md` + update the index |

## Benchmarking

```bash
kbench bench                      # full sweep — lock in final numbers
kbench bench --stride 2           # ~1/2 the workloads — default per iteration
kbench bench --quick              # 2 workloads (smallest + largest) — correctness only
kbench bench --json results.json  # also write results to disk
kbench bench --env PROFILE=1      # env vars do NOT inherit into the container; pass them
kbench ab --a experiments/exp_N/<kernel-file>   # paired A/B, both on one VM
```

**Capture the output**: `kbench bench --stride 2 2>&1 | tee bench.log` — `/skill:log-experiment`
copies `bench.log` into the experiment folder, so a run you didn't tee is a run you can't log.

The GPU backend (`local` / `modal` / `fal`) comes from `config.toml`. If a container
crash-loops (fails to boot repeatedly, not just slow), cancel and fix. Don't wait.

After a benchmark, report: pass/fail counts, absolute latency (min / mean / median / max, split
small vs large when both are present), max abs/rel error, reference latency, and the backend.

## Repo layout

- `config.toml` — kernel definition, GPU backend, container image, benchmark settings
- `experiments/exp_N/` — per-experiment: `plan.md?`, kernel snapshot, `result.md`, `bench.log`
- `experiments/summary.md` — master index, one row per experiment
- `experiments/LESSONS.md` — durable cross-experiment findings (append when a lesson recurs)

## Git

Commit after each `/skill:log-experiment` that changed `summary.md`. Tag only when the user asks.
