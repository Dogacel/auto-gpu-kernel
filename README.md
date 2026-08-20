# Auto GPU Kernel 🏆

Autonomous GPU-kernel discovery & optimizer.

[Technical Report](./archive/report.pdf)

Ranked #1 on [MLSys 2026 - FlashInfer AI Kernel Generation Contest](https://mlsys26.flashinfer.ai/) for the _DeepSeek Sparse Attention (DSA)_ track with an average speedup of 34.93x. Submissions can be found at:

| Kernel | Runtime (ms) |
|---|---|
| [dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64](./archive/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64/) — DSA Sparse Attention | 0.010 
| [dsa_topk_indexer_fp8_h64_d128_topk2048_ps64](./archive/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/) — DSA TopK Indexer | 0.016 

## Setup

Copy the `template` directory into a separate folder / git repository to make sure your agents work in an isolated environment.

The kernel agent is compatible with [FlashInfer](https://github.com/flashinfer-ai/flashinfer) format. Benchmarks run through `kbench`, which targets a local GPU or rents one per run from [Modal](https://modal.com/) or [fal](https://fal.ai/) — so you don't need local hardware. Requires [Claude Code CLI](https://use-claude.com/index).

```bash
uv venv --python 3.12          # creates .venv; VS Code picks it up automatically
source .venv/bin/activate
uv pip install -e ".[modal]"   # or [fal], or [local] on a Linux box with a GPU
```

Solution sources are packed **inside** the GPU container, so the machine driving the agent
needs no CUDA stack — macOS included. The `local` extra pulls `flashinfer-bench`, which is
Linux-only and required only by the `local` backend.

Then get the trace set to wherever your kernel will run:

```bash
# modal
modal setup
modal volume create flashinfer-trace
modal volume put flashinfer-trace /path/to/flashinfer-trace/

# fal — one shared /data per account, so it lives in a subdirectory
fal auth login
fal files upload /path/to/flashinfer-trace flashinfer-trace
# then set [remote.data].path = "/data/flashinfer-trace" in config.toml

# local GPU — just point at it
# [remote.data].local_path = "~/flashinfer-trace"
```

Pick the backend in `config.toml` (`[remote].backend`), or override per-run with
`KBENCH_BACKEND=fal`. `configs/` has ready-made configs for both contest kernels — copy one
over `template/config.toml` to switch targets.

> [!WARNING]
> Absolute latencies are only comparable within one backend. A number from `modal` and a
> number from `fal` are different measurements; `kbench ab` refuses to compare across them.

To get started clone the [MLSys-2026 Contest Dataset](https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest). To change the kernel you are implementing, please refer to the [FlashInfer-Trace - Bring Your Own Kernel](https://bench.flashinfer.ai/docs/tutorials/bring-your-own-kernel) guide.

> [!IMPORTANT]  
> Make sure you update `CLAUDE.md` to describe the kernel you are optimizing. The example in template is customized for sparse attention. Also `optimize.md` and `benchmark.md` has some parameters tuned for sparse attention such as number of test cases to run to get a sanity check. You can ask an agent to help you adjsut them.

## Launch the loop

To run one iteration,

```bash
claude --dangerously-skip-permissions -p "/optimize"
```

Or you can launch interactive mode by running `claude --dangerously-skip-permissions`, selecting the right model, thinking mode and enter `/loop Run /optimize every 15 minutes`.

That's it. The loop runs indefinitely, each iteration picks one optimization, benchmarks it, logs an experiment folder, and continues. Stop with `Ctrl+C` when you want to step in. As agent struggles to find new optimizations, it will start to change its schedule to be less frequent.

## Generic task mode

Besides flashinfer-format kernels, `kopt`/`kbench` can optimize **any repository** with a
self-descriptive task spec — a single TOML that names the repo, the work branch, the bench
commands, the hardware, and the correctness contract. The spec becomes the project's
`config.toml` verbatim; its `description` flows into the agent's AGENTS.md, so everything
the agent needs to know about the target lives in one file you write:

```toml
[task]
name = "my-model"
workdir = "repo"
metric = "seconds for the full generation (lower is better)"
description = """what the repo is, the entry-point contract the verifier pins,
what the correctness gate means, hardware facts, known traps..."""

[task.repo]
url = "git@github.com:org/my-model.git"
branch = "me/auto-optimize"     # created from `base` and pushed if missing

[task.hardware]
gpus = 8
gpu = "H100"

[[task.pinned]]                  # harness-owned verifier, copied in fresh each run
src = "verify/e2e.py"
dst = ".kbench/verify.py"
from_repo = "tools/e2e.py"       # snapshotted from the pristine clone at init

[task.bench.quick]               # any number of named modes
cmd = "uv run .kbench/verify.py --steps 4 --ledger {out}/ledger.json"
artifact = "ledger.json"         # compared byte-for-byte against the golden
golden = "golden/quick.json"
step_regex = 'step \d+ .* (?P<ms>[0-9.]+)ms'
```

```bash
kopt init-task ~/proj task.toml        # clone + work branch + scaffold
cd ~/proj
kbench bench --quick --capture-golden  # pin correctness on the pristine repo
kbench bench --mode full --capture-golden
kopt run ~/proj -n 100 --thinking max
```

Task mode keeps the same principles as kernel mode — absolute numbers, one optimization per
iteration, every experiment logged, no benchmark gaming — but the benchmark is the target
repo's own verification entry point, run locally on all configured GPUs:

- **Pinned verifier**: `[[task.pinned]]` files are harness-owned copies placed fresh into
  the working tree before every run, so agent edits can't change what "correct" means.
- **Golden artifacts**: a mode passes only if its command exits 0 *and* the artifact it
  writes (e.g. a per-step digest ledger) exactly equals the golden captured from the
  unmodified repo (`--capture-golden`).
- **Modes** are free-form (`[task.bench.<name>]`): typically a few-step `quick` run as the
  numerical-correctness gate, a `perf` timing loop, and a `full` end-to-end run whose
  seconds are the metric of record.
- `kbench ab --a <git-ref>` benchmarks a past commit in a temporary worktree back-to-back
  with the current tree on the same machine.
- The agent commits each logged experiment to the target repo and pushes to the configured
  work branch.

## Architecture

For more details on the agentic loop, please refer to the technical report.

Agents:
- Profiler
- Research
- Workload inspector

| Command | Purpose |
|---|---|
| `/optimize` | Main loop |
| `/benchmark <quick\|stride N\|full>` | One-shot benchmark on the configured backend |
| `/log-experiment` | Snapshot + write `result.md` + update index |

See `CLAUDE.md` for rules and `.claude/commands/` for full command specs.


- `solution/triton/solution_fused.py` — the kernel being optimized (overwritten each iteration)
- `config.toml` — kernel definition, GPU backend, container image, benchmark settings
- `experiments/exp_N/` — snapshot + results for iteration N
- `experiments/summary.md` — master index, one row per iteration
- `experiments/LESSONS.md` — durable cross-experiment findings

Past contest submissions, the technical report, and the Fable result set live in `archive/`.

