"""FlashInfer-bench harness.

`collect_sources` runs locally and only reads text files — no flashinfer_bench import, so
the driving machine needs no GPU stack (its wheels are Linux/Windows only). Packing and
evaluation both happen inside the container, which has flashinfer_bench installed.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from kbench.job import Job
from kbench.results import WorkloadResult

SOURCE_SUFFIXES = {".py", ".cu", ".cuh", ".cpp", ".h", ".hpp", ".txt", ".json", ".toml"}


def collect_sources(cfg, override: Path | None = None) -> dict[str, str]:
    """Read the source tree into {relative_path: text}.

    override swaps in a different kernel file (for A/B against a past experiment).
    """
    src = cfg.sources
    if not src.exists():
        raise SystemExit(f"source dir not found: {src}")

    files = {
        str(p.relative_to(src)): p.read_text()
        for p in sorted(src.rglob("*"))
        if p.is_file() and p.suffix in SOURCE_SUFFIXES and not p.name.startswith(".")
    }
    if not files:
        raise SystemExit(f"no source files in {src}")

    if override is not None:
        override = Path(override)
        if not override.exists():
            raise SystemExit(f"override kernel not found: {override}")
        files[cfg.kernel.name] = override.read_text()

    return files


def _pack(job: Job, sources: dict[str, str], tmp: str):
    from flashinfer_bench import BuildSpec
    from flashinfer_bench.agents import pack_solution_from_files
    from flashinfer_bench.data import SupportedBindings

    root = Path(tmp) / "src"
    for name, text in sources.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    spec = BuildSpec(
        language=job.language,
        target_hardware=["cuda"],
        entry_point=job.entry_point,
        binding=SupportedBindings.TVM_FFI if job.language == "cuda" else None,
    )
    # name/author are flashinfer-bench submission metadata; nothing here reads them back.
    return pack_solution_from_files(
        path=str(root),
        spec=spec,
        name=job.definition,
        definition=job.definition,
        author="kbench",
    )


def evaluate(job: Job) -> dict[str, list[WorkloadResult]]:
    """Runs in the GPU container. One entry per candidate label."""
    import logging

    from flashinfer_bench import Benchmark, BenchmarkConfig, TraceSet

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("kbench")

    for key, value in job.env.items():
        os.environ[key] = str(value)
        log.info("env: %s=%s", key, value)

    trace_set = TraceSet.from_path(job.data_path)
    if job.definition not in trace_set.definitions:
        raise ValueError(f"definition {job.definition!r} not in trace set at {job.data_path}")

    definition = trace_set.definitions[job.definition]
    workloads = job.select(trace_set.workloads.get(job.definition, []))
    if not workloads:
        raise ValueError(f"no workloads for {job.definition!r}")
    log.info("running %d workload(s)", len(workloads))

    config = BenchmarkConfig(
        warmup_runs=job.warmup_runs,
        iterations=job.iterations,
        num_trials=job.num_trials,
        # The reference is trivial; profiling it burns GPU time for nothing.
        profile_baseline=job.profile_baseline,
    )

    out: dict[str, list[WorkloadResult]] = {}
    for label, sources in job.sources.items():
        with tempfile.TemporaryDirectory() as tmp:
            solution = _pack(job, sources, tmp)

        bench_set = TraceSet(
            root=trace_set.root,
            definitions={definition.name: definition},
            solutions={definition.name: [solution]},
            workloads={definition.name: workloads},
            traces={definition.name: []},
        )
        traces = (
            Benchmark(bench_set, config)
            .run_all(dump_traces=True)
            .traces.get(definition.name, [])
        )

        results = []
        for trace in traces:
            ev = trace.evaluation
            if not ev:
                continue
            if ev.status.value != "passed" and ev.log:
                log.info("FAIL [%s]\n%s", trace.workload.uuid[:8], ev.log)
            results.append(
                WorkloadResult(
                    workload_id=trace.workload.uuid,
                    status=ev.status.value,
                    latency_ms=ev.performance.latency_ms if ev.performance else None,
                    reference_latency_ms=(
                        ev.performance.reference_latency_ms if ev.performance else None
                    ),
                    max_abs_error=ev.correctness.max_absolute_error if ev.correctness else None,
                    max_rel_error=ev.correctness.max_relative_error if ev.correctness else None,
                    log=ev.log if ev.status.value != "passed" else None,
                )
            )
        out[label] = results

    return out
