"""Run on a local GPU. No container, no upload — the fast path when you have hardware."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from kbench.job import Job
from kbench.results import WorkloadResult


def run(cfg, job: Job) -> dict[str, list[WorkloadResult]]:
    from kbench.flashinfer import evaluate

    trace = Path(cfg.local_path)
    if not trace.exists():
        raise SystemExit(
            f"trace set not found at {trace}\nset [remote.data].local_path in config.toml"
        )

    # Image env is baked into the image on remote backends; apply it by hand here.
    return evaluate(
        replace(job, data_path=str(trace), env={**cfg.image.env, **job.env})
    )
