"""Benchmark adapters describe what to evaluate; backends describe where it runs."""

from __future__ import annotations

from kbench.adapters.base import BenchmarkAdapter, Measurement, RunRequest
from kbench.config import Config, TaskConfig

__all__ = ["BenchmarkAdapter", "Measurement", "RunRequest", "get"]


def get(cfg: Config | TaskConfig) -> BenchmarkAdapter:
    if isinstance(cfg, TaskConfig):
        from kbench.adapters.generated import GeneratedTaskAdapter

        return GeneratedTaskAdapter(cfg)
    from kbench.adapters.flashinfer import FlashInferAdapter

    return FlashInferAdapter(cfg)
