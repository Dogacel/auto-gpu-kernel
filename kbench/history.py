"""Append-only benchmark history.

Every `kbench bench` writes one line here. This is the authoritative optimization
timeline: it comes from BenchResult, not from whatever the agent chose to write into
summary.md, so a chart drawn from it cannot drift from what was actually measured.
"""

from __future__ import annotations

import hashlib
import json
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
