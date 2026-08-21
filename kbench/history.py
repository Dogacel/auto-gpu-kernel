"""Append-only benchmark history.

Every `kbench bench` writes one line here. This is the authoritative optimization
timeline: it comes from BenchResult, not from whatever the agent chose to write into
summary.md, so a chart drawn from it cannot drift from what was actually measured.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from pathlib import Path


def history_path(root: Path) -> Path:
    return Path(root) / ".kopt" / "bench.jsonl"


def kernel_digest(path: Path) -> str:
    """Short content hash, so a measurement can be tied to an exact kernel version."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
    except OSError:
        return ""


def record(cfg, result, mode: str) -> None:
    """Append one measurement. Never raises into a benchmark run."""
    try:
        summary = result.summary()
        entry = {
            "t": time.time(),
            "mode": mode,
            "kernel": kernel_digest(cfg.kernel),
            **summary,
        }
        path = history_path(cfg.root)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def record_measurement(root: Path, result) -> None:
    """Record the common adapter result shape. Never raises into a benchmark run."""
    try:
        samples = result.samples
        count = max(len(samples), 1)
        entry = {
            "t": time.time(),
            "mode": result.mode,
            "kernel": result.candidate_rev,
            "harness": result.harness_rev,
            "definition": result.adapter,
            "backend": result.adapter,
            "gpu": result.provenance,
            "num_workloads": count,
            "num_passed": count if result.passed else 0,
            "passed": result.passed,
            "validation": result.validation,
            "metric_value": result.value,
            "metric": result.unit,
            "lower_is_better": result.lower_is_better,
            "samples": samples,
            "sample_mean": statistics.fmean(samples) if samples else None,
            "sample_median": statistics.median(samples) if samples else None,
            "sample_min": min(samples) if samples else None,
            "sample_max": max(samples) if samples else None,
        }
        path = history_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def load(root: Path) -> list[dict]:
    path = history_path(root)
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out
