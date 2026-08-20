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

