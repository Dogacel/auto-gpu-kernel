"""Built-in adapter for FlashInfer benchmark definitions and trace sets."""

from __future__ import annotations

from pathlib import Path

from kbench import backends
from kbench.adapters.base import BenchmarkAdapter, Measurement, RunRequest
from kbench.config import Config
from kbench.flashinfer import collect_sources
from kbench.history import kernel_digest, record
from kbench.job import Job
from kbench.results import BenchResult, print_ab, print_results


class FlashInferAdapter(BenchmarkAdapter):
    name = "flashinfer"

    def __init__(self, cfg: Config):
        super().__init__(cfg.root)
        self.cfg = cfg

    def _job(self, sources: dict[str, dict[str, str]], request: RunRequest) -> Job:
        if request.mode not in {"quick", "stride", "full"}:
            raise SystemExit(f"FlashInfer does not support mode {request.mode!r}")
        bench = self.cfg.bench
        return Job(
            definition=self.cfg.definition,
            language=self.cfg.language,
            entry_point=self.cfg.entry_point,
            sources=sources,
            data_path=self.cfg.data_path,
            mode=request.mode,
            stride=request.stride,
            warmup_runs=bench.get("warmup_runs", 3),
            iterations=bench.get("iterations", 100),
            num_trials=bench.get("num_trials", 5),
            profile_baseline=bench.get("profile_baseline", False),
            env=request.env,
        )

    def run(
        self,
        candidates: dict[str, str | None],
        request: RunRequest,
    ) -> dict[str, Measurement]:
        source_sets: dict[str, dict[str, str]] = {}
        paths: dict[str, Path] = {}
        for label, selector in candidates.items():
            path = Path(selector).resolve() if selector is not None else self.cfg.kernel
            paths[label] = path
            source_sets[label] = collect_sources(
                self.cfg,
                override=path if selector is not None else None,
            )

        job = self._job(source_sets, request)
        raw = backends.get(self.cfg.backend).run(self.cfg, job)
        measurements = {}
        for label, workloads in raw.items():
            native = BenchResult(
                definition=self.cfg.definition,
                backend=self.cfg.backend,
                gpu=self.cfg.gpu,
                workloads={workload.workload_id: workload for workload in workloads},
            )
            summary = native.summary()
            total = len(native.workloads)
            measurements[label] = Measurement(
                adapter=self.name,
                label=label,
                candidate=str(paths[label]),
                mode=job.mode if job.mode != "stride" else f"stride{job.stride}",
                passed=total > 0 and native.n_passed == total,
                value=summary["mean_latency_ms"],
                unit="ms",
                lower_is_better=True,
                samples=native.latencies,
                validation=f"{native.n_passed}/{total} workloads passed",
                provenance=native.provenance,
                candidate_rev=kernel_digest(paths[label]),
                harness_rev="built-in-flashinfer",
                native=native,
            )
        return measurements

    def print_result(self, result: Measurement) -> None:
        print_results(result.native)

    def print_ab(self, a: Measurement, b: Measurement) -> None:
        print_ab(a.native, b.native, a.candidate, b.candidate)

    def record(self, result: Measurement) -> None:
        record(self.cfg, result.native, result.mode)

    def serialize(self, result: Measurement) -> dict:
        return result.native.to_dict()
