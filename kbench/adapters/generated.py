"""Adapter bridge for an agent-generated validate.py and benchmark.py pair."""

from __future__ import annotations

from dataclasses import asdict

from kbench import task
from kbench.adapters.base import BenchmarkAdapter, Measurement, RunRequest
from kbench.config import TaskConfig


class GeneratedTaskAdapter(BenchmarkAdapter):
    name = "generated-task"

    def __init__(self, cfg: TaskConfig):
        super().__init__(cfg.root)
        self.cfg = cfg

    def _measurement(
        self,
        label: str,
        candidate: str,
        result: task.TaskResult,
    ) -> Measurement:
        return Measurement(
            adapter=self.name,
            label=label,
            candidate=candidate,
            mode=result.mode,
            passed=result.passed,
            value=result.value,
            unit=result.unit,
            lower_is_better=result.lower_is_better,
            samples=result.samples,
            validation=result.validation_status,
            provenance=f"local / {self.cfg.gpu} x{self.cfg.gpus}",
            candidate_rev=result.workdir_rev,
            harness_rev=result.harness_rev,
            artifacts=result.out_dir,
            native=result,
        )

    def run(
        self,
        candidates: dict[str, str | None],
        request: RunRequest,
    ) -> dict[str, Measurement]:
        if request.mode not in task.MODES:
            raise SystemExit("generated task adapters support only quick and full modes")
        items = list(candidates.items())
        if len(items) == 1 and items[0][1] is None:
            label, _ = items[0]
            result = task.run_mode(self.cfg, request.mode, extra_env=request.env)
            return {label: self._measurement(label, "current tree", result)}
        if len(items) == 2 and items[0][1] is not None and items[1][1] is None:
            (a_label, baseline), (b_label, _) = items
            a, b = task.run_ab(
                self.cfg,
                baseline,
                request.mode,
                extra_env=request.env,
            )
            return {
                a_label: self._measurement(a_label, baseline, a),
                b_label: self._measurement(b_label, "current tree", b),
            }
        raise SystemExit(
            "generated task adapter expects the current tree, or one git ref plus current"
        )

    def print_result(self, result: Measurement) -> None:
        task.print_result(self.cfg, result.native)

    def print_ab(self, a: Measurement, b: Measurement) -> None:
        task.print_ab(self.cfg, a.native, b.native, a.candidate)

    def record(self, result: Measurement) -> None:
        task.record(self.cfg, result.native)

    def serialize(self, result: Measurement) -> dict:
        return asdict(result.native)
