"""Normalized benchmark results. Everything downstream consumes only this."""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field


@dataclass
class WorkloadResult:
    workload_id: str
    status: str  # "passed" | "failed" | "error"
    latency_ms: float | None = None
    reference_latency_ms: float | None = None
    max_abs_error: float | None = None
    max_rel_error: float | None = None
    log: str | None = None

    @property
    def passed(self) -> bool:
        return self.status.lower() == "passed"


@dataclass
class BenchResult:
    definition: str
    backend: str
    gpu: str
    workloads: dict[str, WorkloadResult] = field(default_factory=dict)

    # --- provenance -----------------------------------------------------
    @property
    def provenance(self) -> str:
        """Absolute latencies are only comparable within one backend+GPU."""
        return f"{self.backend} / {self.gpu}"

    # --- aggregates -----------------------------------------------------
    @property
    def latencies(self) -> list[float]:
        return [
            w.latency_ms
            for w in self.workloads.values()
            if w.latency_ms is not None and not math.isnan(w.latency_ms)
        ]

    @property
    def n_passed(self) -> int:
        return sum(1 for w in self.workloads.values() if w.passed)

    def summary(self) -> dict:
        lat = self.latencies
        finite = lambda xs: [x for x in xs if x is not None and not math.isnan(x)]
        abs_errs = finite([w.max_abs_error for w in self.workloads.values()])
        rel_errs = finite([w.max_rel_error for w in self.workloads.values()])
        return {
            "definition": self.definition,
            "backend": self.backend,
            "gpu": self.gpu,
            "num_workloads": len(self.workloads),
            "num_passed": self.n_passed,
            "mean_latency_ms": statistics.fmean(lat) if lat else None,
            "std_latency_ms": statistics.stdev(lat) if len(lat) > 1 else 0.0 if lat else None,
            "median_latency_ms": statistics.median(lat) if lat else None,
            "min_latency_ms": min(lat) if lat else None,
            "max_latency_ms": max(lat) if lat else None,
            "worst_max_abs_error": max(abs_errs) if abs_errs else None,
            "worst_max_rel_error": max(rel_errs) if rel_errs else None,
        }

    def to_dict(self) -> dict:
        return {
            "summary": self.summary(),
            "workloads": {k: asdict(v) for k, v in self.workloads.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BenchResult":
        s = d["summary"]
        return cls(
            definition=s["definition"],
            backend=s["backend"],
            gpu=s["gpu"],
            workloads={
                k: WorkloadResult(**v) for k, v in d["workloads"].items()
            },
        )


def _fmt(v: float | None, spec: str = ".3f") -> str:
    # NaN is a distinct signal (the kernel produced NaNs, not merely inaccurate
    # results), and it also poisons min/max. Never render it as a number.
    if v is None:
        return "-"
    if math.isnan(v):
        return "NaN"
    return format(v, spec)


def print_results(r: BenchResult) -> None:
    print(f"\n{r.definition}  [{r.provenance}]")
    for uid, w in r.workloads.items():
        line = f"  {uid[:8]}  {w.status:8}"
        if w.latency_ms is not None:
            line += f" {w.latency_ms:9.3f} ms"
        # 0 means the baseline was not profiled (bench.profile_baseline = false),
        # not that the reference took no time. Don't render it as a measurement.
        if w.reference_latency_ms:
            line += f"  ref {w.reference_latency_ms:8.3f} ms"
        if w.max_abs_error is not None:
            line += f"  abs {_fmt(w.max_abs_error, '.2e')} rel {_fmt(w.max_rel_error, '.2e')}"
        print(line)
        if not w.passed and w.log:
            print(f"      {w.log.strip().splitlines()[-1][:200]}")

    s = r.summary()
    print(
        f"\n  passed {s['num_passed']}/{s['num_workloads']}"
        f" | mean {_fmt(s['mean_latency_ms'])} ms"
        f" | median {_fmt(s['median_latency_ms'])} ms"
        f" | min {_fmt(s['min_latency_ms'])} ms"
        f" | max {_fmt(s['max_latency_ms'])} ms"
    )
    # A passing workload with a huge relative error is near-zero-reference inflation,
    # not a correctness problem. Say so, or the agent burns iterations chasing it.
    inflated = [
        w for w in r.workloads.values()
        if w.passed and w.max_rel_error is not None and w.max_rel_error > 1.0
    ]
    if inflated:
        print(
            f"  note: {len(inflated)} passing workload(s) show relative error > 1 —"
            " near-zero reference values inflate the ratio; they met the harness"
            " tolerance. Judge correctness on pass/fail and absolute error."
        )
    if s["worst_max_abs_error"] is not None:
        print(
            f"  worst abs err {s['worst_max_abs_error']:.2e}"
            f" | worst rel err {_fmt(s['worst_max_rel_error'], '.2e')}"
        )


def print_ab(a: BenchResult, b: BenchResult, a_label: str, b_label: str) -> None:
    """Paired comparison. Same VM, same workloads — the only trustworthy delta."""
    if (a.backend, a.gpu) != (b.backend, b.gpu):
        raise SystemExit(
            f"refusing to compare across backends: {a.provenance} vs {b.provenance}"
        )

    print(f"\nA = {a_label}\nB = {b_label}   [{a.provenance}]\n")
    print(f"{'workload':10} {'A (ms)':>10} {'B (ms)':>10} {'B-A':>11} {'%':>8}  win")
    deltas = []
    for uid in sorted(a.workloads):
        wa, wb = a.workloads.get(uid), b.workloads.get(uid)
        if not wa or not wb or wa.latency_ms is None or wb.latency_ms is None:
            continue
        d = wb.latency_ms - wa.latency_ms
        pct = 100.0 * d / wa.latency_ms if wa.latency_ms else 0.0
        deltas.append(d)
        print(
            f"{uid[:8]:10} {wa.latency_ms:10.4f} {wb.latency_ms:10.4f} "
            f"{d:+11.4f} {pct:+7.2f}%  {'B' if d < 0 else 'A' if d > 0 else '='}"
        )

    if not deltas:
        print("\n  no comparable workloads")
        return
    n_b = sum(1 for d in deltas if d < 0)
    mean = statistics.fmean(deltas)
    verdict = "B faster" if mean < 0 else "A faster" if mean > 0 else "tie"
    print(f"\n  paired n={len(deltas)} | B wins {n_b}/{len(deltas)} | mean {mean:+.4f} ms -> {verdict}")
    if deltas and abs(mean) < 0.05 * statistics.fmean(
        [w.latency_ms for w in a.workloads.values() if w.latency_ms]
    ):
        print("  NOTE: delta is under 5% — treat as noise unless it reproduces")
