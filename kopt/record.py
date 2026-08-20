"""Append-only run log.

The loop records; it does not render. Everything the agent emits lands in one NDJSON
file, and any number of readers (`kopt watch`, `tail -f`, a later analysis script) consume
it independently. Nothing in the loop knows a viewer exists.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any


def runs_dir(project: Path) -> Path:
    return Path(project) / ".kopt" / "runs"


def new_run_log(project: Path) -> Path:
    """A fresh file per run, so finished runs stay inspectable."""
    return runs_dir(project) / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl"


def list_run_logs(project: Path) -> list[Path]:
    d = runs_dir(project)
    return sorted(d.glob("*.jsonl")) if d.is_dir() else []


def latest_run_log(project: Path) -> Path | None:
    runs = list_run_logs(project)
    return runs[-1] if runs else None


def _plain(value: Any) -> Any:
    """Best-effort JSON-able view of an omp-rpc event dataclass."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _plain(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class Recorder:
    """Writes one JSON object per line. Never raises into the loop."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()

    def write(self, kind: str, **fields) -> None:
        record = {"t": time.time(), "kind": kind, **fields}
        try:
            with self.path.open("a") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass  # observability must never take down a run

    def attach(self, client) -> None:
        """Mirror every agent event into the log."""
        client.on_event(
            lambda event: self.write(
                "event",
                type=getattr(event, "type", type(event).__name__),
                data=_plain(event),
            )
        )
