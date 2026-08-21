"""Common contract for repository-specific benchmark implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import statistics
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunRequest:
    """Kbench-owned controls passed to every benchmark adapter."""

    mode: str
    stride: int = 1
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class Measurement:
    """The common result surface returned by every adapter."""

    adapter: str
    label: str
    candidate: str
    mode: str
    passed: bool
    value: float | None
    unit: str
    lower_is_better: bool
    samples: list[float]
    validation: str
    provenance: str
    candidate_rev: str = ""
    harness_rev: str = ""
    artifacts: str = ""
    native: Any = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "label": self.label,
            "candidate": self.candidate,
            "mode": self.mode,
            "passed": self.passed,
            "value": self.value,
            "unit": self.unit,
            "lower_is_better": self.lower_is_better,
            "samples": self.samples,
            "validation": self.validation,
            "provenance": self.provenance,
            "candidate_rev": self.candidate_rev,
            "harness_rev": self.harness_rev,
            "artifacts": self.artifacts,
        }


class BenchmarkAdapter(ABC):
    """A benchmark implementation for one project type.

    Candidate values are adapter-owned selectors: FlashInfer receives a source file,
    while a generated repository adapter receives a git ref. ``None`` always means the
    current candidate.
    """

    name: str

    def __init__(self, root: Path):
        self.root = Path(root)

    @abstractmethod
    def run(
        self,
        candidates: dict[str, str | None],
        request: RunRequest,
    ) -> dict[str, Measurement]:
        """Evaluate candidates, preserving paired execution when more than one is given."""

    def bench(self, request: RunRequest) -> Measurement:
        return self.run({"main": None}, request)["main"]

    def ab(self, baseline: str, request: RunRequest) -> tuple[Measurement, Measurement]:
        results = self.run({"a": baseline, "b": None}, request)
        return results["a"], results["b"]

    def print_result(self, result: Measurement) -> None:
        """Render a normalized result; adapters may override with richer detail."""
        status = "PASS" if result.passed else "FAIL"
        value = f"{result.value:.6g} {result.unit}" if result.value is not None else "-"
        print(
            f"\n{result.adapter}  [{result.mode}]  {result.provenance}"
            f"  candidate={result.candidate_rev or result.candidate}"
        )
        print(f"  {status}  validation={result.validation}  metric={value}")
        if result.samples:
            print(
                f"  samples n={len(result.samples)}: min={min(result.samples):.6g}"
                f" mean={statistics.fmean(result.samples):.6g}"
                f" max={max(result.samples):.6g}"
            )
        if result.artifacts:
            print(f"  artifacts: {result.artifacts}")

    def print_ab(self, a: Measurement, b: Measurement) -> None:
        """Render a normalized paired comparison."""
        print(f"\nA = {a.candidate}\nB = {b.candidate}   [{b.provenance}]\n")
        for result in (a, b):
            value = (
                f"{result.value:.6g} {result.unit}"
                if result.value is not None
                else "-"
            )
            print(f"  {result.label.upper()}: {'PASS' if result.passed else 'FAIL'}  {value}")
        if not self.comparable(a, b) or a.value is None or b.value is None or a.value == 0:
            print("\n  no comparable measurements")
            return
        delta = b.value - a.value
        pct = 100.0 * delta / abs(a.value)
        b_better = delta < 0 if b.lower_is_better else delta > 0
        verdict = "B better" if b_better else "A better" if delta else "tie"
        print(f"\n  B-A = {delta:+.6g} {b.unit} ({pct:+.2f}%) -> {verdict}")

    def record(self, result: Measurement) -> None:
        """Append the normalized benchmark history entry."""
        from kbench.history import record_measurement

        record_measurement(self.root, result)

    def serialize(self, result: Measurement) -> dict[str, Any]:
        return result.to_dict()

    @staticmethod
    def comparable(a: Measurement, b: Measurement) -> bool:
        return (
            a.passed
            and b.passed
            and a.provenance == b.provenance
            and a.harness_rev == b.harness_rev
            and a.unit == b.unit
            and a.lower_is_better == b.lower_is_better
        )
