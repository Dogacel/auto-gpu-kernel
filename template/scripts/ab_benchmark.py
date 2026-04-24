"""
Paired A/B benchmark on the same Modal VM — eliminates cross-VM reference
latency variance (20-30%) when comparing two candidate kernels.

Usage:
    modal run scripts/ab_benchmark.py::run --a experiments/exp_N/sparse_fused.py

A is the baseline (e.g., previous best). B is always the current
solution/triton/sparse_fused.py. Both are packed identically and run
back-to-back in one Modal function call → one container → one GPU.

Reports per-workload paired delta (B − A) and aggregate win rate.
"""

import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

from flashinfer_bench import Benchmark, BenchmarkConfig, BuildSpec, Solution, TraceSet
from flashinfer_bench.agents import pack_solution_from_files

# Separate app from run_modal's so entrypoint names don't collide.
app = modal.App("flashinfer-ab")

# Modal uploads only the entrypoint script to the container — importing from
# scripts.run_modal fails remotely. So duplicate the image + volume here.
# KEEP IN SYNC WITH scripts/run_modal.py image definition.
TRACE_SET_PATH = "/data"
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "git-lfs", "wget", "build-essential", "cmake")
    .pip_install(
        "torch",
        extra_index_url="https://download.pytorch.org/whl/cu130",
    )
    .pip_install(
        "apache-tvm-ffi>=0.1.6,!=0.1.8,!=0.1.8.post0,<0.2",
        "click",
        "einops",
        "ninja",
        "numpy",
        "nvidia-cudnn-frontend>=1.13.0",
        "nvidia-cutlass-dsl>=4.3.4",
        "nvidia-ml-py",
        "packaging>=24.2",
        "requests",
        "tabulate",
        "tqdm",
        "responses",
        "pytest",
        "scipy",
        "build",
        "cuda-python==13.0",
        "nvidia-cudnn-cu13>=9.14.0.64",
        "flashinfer-bench @ git+https://github.com/flashinfer-ai/flashinfer-bench.git@80f40d45968c65840d05872516befd9691ec9fd8",
        "tilelang",
        "cuda-tile",
        "cupti-python",
        "pandas",
        "cupy-cuda13x",
    )
    .env({
        "LD_LIBRARY_PATH": "/opt/conda/envs/py312/lib/python3.12/site-packages/nvidia/cu13/lib/",
        "TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas",
    })
)

TRITON_DIR = PROJECT_ROOT / "solution" / "triton"
DEFINITION = "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64"


def _pack(candidate_py: Path, tag: str) -> Solution:
    """Pack a Solution using candidate_py as sparse_fused.py."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "triton"
        src.mkdir()
        shutil.copy(TRITON_DIR / "sparse_baseline.py", src / "sparse_baseline.py")
        shutil.copy(candidate_py, src / "sparse_fused.py")
        spec = BuildSpec(
            language="triton",
            target_hardware=["cuda"],
            entry_point="sparse_fused.py::kernel",
            binding=None,
        )
        return pack_solution_from_files(
            path=str(src),
            spec=spec,
            name=f"ab-{tag}",
            definition=DEFINITION,
            author="ab_benchmark",
        )


@app.function(image=image, gpu="B200:1", timeout=2400, retries=0, volumes={TRACE_SET_PATH: trace_volume})
def run_ab(sol_a: Solution, sol_b: Solution, stride: int = 2) -> dict:
    config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5, profile_baseline=False)
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition = trace_set.definitions[DEFINITION]
    workloads = trace_set.workloads.get(DEFINITION, [])
    if stride > 1:
        workloads = workloads[::stride]

    out = {"a": {}, "b": {}, "ref": {}}
    for side, sol in (("a", sol_a), ("b", sol_b)):
        bts = TraceSet(
            root=trace_set.root,
            definitions={definition.name: definition},
            solutions={definition.name: [sol]},
            workloads={definition.name: workloads},
            traces={definition.name: []},
        )
        res = Benchmark(bts, config).run_all(dump_traces=True)
        for t in res.traces.get(definition.name, []):
            if t.evaluation and t.evaluation.performance:
                out[side][t.workload.uuid] = t.evaluation.performance.latency_ms
                out["ref"][t.workload.uuid] = t.evaluation.performance.reference_latency_ms
    return out


@app.local_entrypoint()
def run(a: str, stride: int = 2):
    a_path = Path(a).resolve()
    b_path = TRITON_DIR / "sparse_fused.py"
    assert a_path.exists(), f"A-side kernel not found: {a_path}"

    print(f"A (baseline): {a_path}")
    print(f"B (current):  {b_path}")
    print(f"Stride:       {stride}\n")

    sol_a = _pack(a_path, "a")
    sol_b = _pack(b_path, "b")

    out = run_ab.remote(sol_a, sol_b, stride=stride)

    print(f"{'UUID':10} {'A (ms)':>10} {'B (ms)':>10} {'ΔB−A (ms)':>12} {'%':>8}  winner")
    deltas = []
    for uuid in sorted(out["a"]):
        a_ms, b_ms = out["a"].get(uuid), out["b"].get(uuid)
        if a_ms is None or b_ms is None:
            continue
        d = b_ms - a_ms
        pct = 100.0 * d / a_ms if a_ms else 0.0
        deltas.append(d)
        w = "B" if d < 0 else ("A" if d > 0 else "=")
        print(f"{uuid[:8]:10} {a_ms:10.4f} {b_ms:10.4f} {d:+12.4f} {pct:+7.2f}%  {w}")

    if deltas:
        n = len(deltas)
        n_b = sum(1 for d in deltas if d < 0)
        mean = sum(deltas) / n
        print(f"\nPaired n={n} | B wins {n_b}/{n} | mean Δ = {mean:+.4f} ms "
              f"→ {'B faster' if mean < 0 else 'A faster' if mean > 0 else 'tie'}")
