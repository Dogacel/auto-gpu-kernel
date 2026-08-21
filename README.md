# Auto GPU Kernel 🏆

Autonomous GPU-kernel discovery & optimizer.

[Technical Report](./archive/report.pdf)

Ranked #1 on [MLSys 2026 - FlashInfer AI Kernel Generation Contest](https://mlsys26.flashinfer.ai/) for the _DeepSeek Sparse Attention (DSA)_ track with an average speedup of 34.93x. Submissions + optimized kernels can be found at [archive](./archive/).

`kopt` runs the optimization agent. `kbench` owns validation, benchmarks, result
history, and paired A/B through a small adapter interface.

## Install

You need Python 3.11+, [uv](https://docs.astral.sh/uv/), and
[OMP](https://github.com/can1357/oh-my-pi).

```bash
uv venv --python 3.12
source .venv/bin/activate

# Arbitrary local repositories
uv pip install -e ".[agent]"

# FlashInfer: choose local on Linux with a GPU, or Modal from any machine
# uv pip install -e ".[agent,local]"
# uv pip install -e ".[agent,modal]"
```

Sign in once:

```bash
omp                 # model provider
```

## Run a FlashInfer kernel

Download a trace set:

```bash
git lfs install
git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest
```

Upload it to Modal:

```bash
modal setup
modal volume create flashinfer-trace
modal volume put flashinfer-trace ./mlsys26-contest/
```

Create a project and run three iterations:

```bash
kopt init ~/topk-run \
  ./mlsys26-contest/definitions/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.json \
  --backend modal --gpu B200

cd ~/topk-run
kbench bench --quick
kopt run . -n 3 --model anthropic/claude-opus-5 --thinking low
```

Useful benchmark commands:

```bash
kbench bench --quick       # smallest and largest workload
kbench bench --stride 2    # half of the workloads
kbench bench               # full run
kbench ab --a experiments/exp_3/solution_fused.py
```

## Run against any Git repository

Write a short `task.toml`. Describe the job in plain language; the setup agent will
inspect the repository and build the benchmark harness.

```toml
[task]
name = "my-project"
workdir = "repo"
objective = """
Speed up inference without changing the public API or model outputs.
The optimizer may change code below src/runtime/.
"""
measure = """
Measure end-to-end latency for the representative example in examples/serve.py.
Lower latency is better. Include warmup and synchronize the GPU before timing.
"""
validate = """
Run the existing correctness tests and compare the example's output with the untouched
repository. Outputs must match exactly.
"""
hints = "Start with allocations and repeated kernel launches in the decode loop."

[task.repo]
url = "git@github.com:my-org/my-project.git"
base = "main"
branch = "me/auto-optimize"

[task.hardware]
gpus = 1
gpu = "H100"
```

There are no pinned-file or benchmark-command sections in this file. The setup agent
implements the generated-task adapter by writing `harness/validate.py` and
`harness/benchmark.py`; kbench decides how they run.

Generated-task adapters currently run on the local machine, so the target repository's
dependencies and any required GPU must be available there. The generated harness is
trusted executable code: kbench records exact revisions and rejects in-run mutation,
but it is not an operating-system sandbox.

Then scaffold and run:

```bash
kopt init-task ~/my-run task.toml
kopt run ~/my-run -n 20 --model anthropic/claude-opus-5 --thinking low
```

On the first run, kopt uses one setup turn to inspect the untouched clone and generate
the validation and benchmark adapters. Kbench then runs pristine quick and full
baselines. The 20 requested optimization iterations begin after setup.

The cloned repository lives inside `~/my-run`; auto-gpu-kernel itself is never used as
the agent's working directory. During setup the clone is read-only. During optimization
the agent can edit the clone or improve the project-local harness. Every result records
both revisions, and A/B always uses the same current harness for A and B. Setup aborts
if the builder changes either the clone or the auto-gpu-kernel checkout.

To inspect or rerun what the setup agent made:

```bash
cd ~/my-run
cat harness/README.md
kbench bench --quick            # quick validation + quick measurement
kbench bench                    # full validation + metric of record
kbench ab --a <git-ref>         # same current harness for A and B
```

## Watch a run

```bash
kopt watch .
```

Open <http://127.0.0.1:8765>.

```text
config.toml          human task brief
harness/validate.py  generated quick/full correctness adapter
harness/benchmark.py generated quick/full measurement adapter
harness/prepared.json pristine baseline metadata
.omp/                project-local agent instructions
.kopt/runs/          agent logs
.kopt/bench.jsonl    benchmark history
experiments/         experiment notes and snapshots
```

## Architecture

FlashInfer and arbitrary repositories use the same `BenchmarkAdapter` lifecycle:

- `FlashInferAdapter` packages kernel sources and sends them through a local or Modal
  execution backend.
- `GeneratedTaskAdapter` runs the repository-specific validation and benchmark scripts
  created by the setup agent.
- Kbench supplies quick/full execution, normalized measurements, history, and paired
  A/B for both adapters.

See [kbench/README.md](./kbench/README.md) and [kopt/README.md](./kopt/README.md) for the
small class diagrams.

## FAL

FAL is experimental and is not part of the v1 supported path. Its current FlashInfer
image uses Python 3.10 while this package requires Python 3.11+, so use local or Modal
for now.
