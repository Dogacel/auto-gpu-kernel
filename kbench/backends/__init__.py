"""Backends all expose: run(cfg, payload) -> {label: {workload_id: result_dict}}."""

from __future__ import annotations

BACKENDS = ("local", "modal", "fal")


def get(name: str):
    if name == "local":
        from . import local

        return local
    if name == "modal":
        from . import modal as modal_backend

        return modal_backend
    if name == "fal":
        from . import fal as fal_backend

        return fal_backend
    raise SystemExit(f"unknown backend {name!r} (expected one of {', '.join(BACKENDS)})")
