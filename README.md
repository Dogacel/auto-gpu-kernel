# Auto GPU Kernel 🏆

Autonomous GPU-kernel discovery & optimizer.

[Technical Report](./report.pdf)

Ranked #1 on [MLSys 2026 - FlashInfer AI Kernel Generation Contest](https://mlsys26.flashinfer.ai/) for the _DeepSeek Sparse Attention (DSA)_ track with an average speedup of 34.93x. Submissions can be found at:

| Kernel | Runtime (ms) | Speedup |
|---|---|---|
| [dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64](./dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64/) — DSA Sparse Attention | 0.010 |
| [dsa_topk_indexer_fp8_h64_d128_topk2048_ps64](./dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/) — DSA TopK Indexer | 0.016 |

## Setup

Copy the `template` directory into a separate folder / git repository to make sure your agents work in an isolated environment.

The kernel agent is compatible with [FlashInfer](https://github.com/flashinfer-ai/flashinfer) format and can run without a local GPU on cloud using [Modal](https://modal.com/). Requires [Claude Code CLI](https://use-claude.com/index).

```bash
# Python env
conda create -n fi-bench python=3.12
conda activate fi-bench
pip install flashinfer-bench modal

# One-time environment setup
modal setup
modal volume create flashinfer-trace
modal volume put flashinfer-trace /path/to/flashinfer-trace/
```

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

Agents:
- Profiler:
- Research:
- Workload inspector:

| Command | Purpose |
|---|---|
| `/optimize` | Main loop |
| `/benchmark <quick\|stride N\|full>` | One-shot Modal run |
| `/log-experiment` | Snapshot + write `result.md` + update index |

See `CLAUDE.md` for rules and `.claude/commands/` for full command specs.

[Describe Loop Here]

- `solution/triton/sparse_fused.py` — the kernel being optimized (overwritten each iteration)
- `experiments/exp_N/` — snapshot + results for iteration N
- `experiments/summary.md` — master index, one row per iteration
- `experiments/LESSONS.md` — durable cross-experiment findings

