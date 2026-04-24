"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest", add_python="3.12")
    .apt_install("git", "wget", "build-essential", "cmake")
    .pip_install("huggingface_hub")
    .run_commands(
        # Force-reinstall latest main of flashinfer-bench. Override the
        # PyPI 0.1.2 already in the image. Pip pulls/refreshes deps as
        # needed (pydantic, safetensors, tvm-ffi, ...).
        "pip install --force-reinstall --upgrade "
        "git+https://github.com/flashinfer-ai/flashinfer-bench.git@main",
    )
    .pip_install(
        # cupti-python is required for accurate GPU timing (per
        # EVALUATION.md). Without it, flashinfer-bench's timer falls
        # back to CUDA events, which include kernel-launch overhead
        # (~2-4 μs per call) and over-report latency. The official
        # image is *supposed* to ship this but the released 0.1.2
        # dep-chain doesn't pull it in.
        "cupti-python",
    )
    # Caches the CUTLASS clone + torch extension build dir across runs.
    .env({
        "TORCH_EXTENSIONS_DIR": "/results/torch_ext_cache",
        "DSA_CUTLASS_DIR": "/results/cutlass",
    })
)

@app.function(image=image, gpu="B200:1", timeout=1800, retries=3, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(solution: Solution, config: BenchmarkConfig = None, max_workloads: int = 0,
                  stride: int = 1, quick: bool = False, env_vars: dict = None) -> dict:
    """Run benchmark on Modal B200 and return results."""
    import logging
    import os
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    # Modal env vars do NOT propagate from the shell — must be set explicitly
    # inside the container. This is the only way to enable e.g. DSA_PROFILE=1.
    if env_vars:
        for k, v in env_vars.items():
            os.environ[k] = str(v)
            logger.info(f"env: {k}={v}")

    if config is None:
        # Don't change profile_baseline=False, the given baseline is too simple and wastes too much time.
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5, profile_baseline=False)

    logger.info(f"Loading trace set from {TRACE_SET_PATH}")
    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if quick:
        # Smallest + largest — catches shape-assumption bugs that a single
        # workload (typically the smallest) would silently pass.
        workloads = [workloads[0], workloads[-1]] if len(workloads) >= 2 else workloads[:1]
        logger.info(f"Quick mode: running {len(workloads)} workload(s) (first + last)")
    elif stride > 1:
        workloads = workloads[::stride]
        logger.info(f"Sampled {len(workloads)} workload(s) (stride={stride})")
    if max_workloads > 0:
        workloads = workloads[:max_workloads]
        logger.info(f"Limited to {len(workloads)} workload(s)")

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.status.value != "passed" and trace.evaluation.log:
                logger.info(f"FULL LOG [{trace.workload.uuid[:8]}]:\n{trace.evaluation.log}")
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()


@app.local_entrypoint()
def main(quick: bool = False, stride: int = 1, env: str = ""):
    """Pack solution and run benchmark on Modal.

    Args:
        quick: Smallest + largest workload (correctness check).
        stride: Sample every Nth workload (default 1 = all).
        env: Extra env vars as K=V,K2=V2 (e.g. "DSA_PROFILE=1,DSA_FOO=bar").
             Modal does NOT inherit shell env vars — this is the only path.
    """
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    env_vars = {}
    if env:
        for pair in env.split(","):
            k, _, v = pair.partition("=")
            if k:
                env_vars[k.strip()] = v.strip()
                print(f"Setting {k.strip()}={v.strip()} inside the Modal container")
    env_vars = env_vars or None

    if quick:
        print("\nRunning benchmark on Modal B200 (quick mode: smallest + largest workload)...")
    elif stride > 1:
        print(f"\nRunning benchmark on Modal B200 (stride={stride})...")
    else:
        print("\nRunning benchmark on Modal B200...")
    results = run_benchmark.remote(solution, stride=stride, quick=quick, env_vars=env_vars)

    if not results:
        print("No results returned!")
        return

    print_results(results)
