"""fal backend. Same ImageSpec, rendered to a Dockerfile.

fal has one shared /data volume per account (no named volumes), so the trace set lives
in a subdirectory — set [remote.data].path to e.g. /data/flashinfer-trace.
Upload with:  fal files upload <local-trace-set> flashinfer-trace
"""

from __future__ import annotations

from dataclasses import replace

from kbench.job import Job
from kbench.results import WorkloadResult

MACHINE_TYPES = {
    "A100": "GPU-A100",
    "H100": "GPU-H100",
    "H200": "GPU-H200",
    "B200": "GPU-B200",
    "L40": "GPU-L40",
    "RTXPRO6000": "GPU-RTXPRO6000",
}


def run(cfg, job: Job) -> dict[str, list[WorkloadResult]]:
    import fal
    from fal.container import ContainerImage

    from kbench.harness import evaluate

    machine = MACHINE_TYPES.get(cfg.gpu.upper().replace("-", ""))
    if machine is None:
        raise SystemExit(
            f"fal has no machine type for gpu={cfg.gpu!r}; "
            f"known: {', '.join(sorted(MACHINE_TYPES))}"
        )

    remote = fal.function(
        kind="container",
        image=ContainerImage.from_dockerfile_str(cfg.image.to_dockerfile()),
        machine_type=machine,
        num_gpus=cfg.gpu_count,
        request_timeout=cfg.timeout_s,
        local_python_modules=["kbench"],
    )(evaluate)

    return remote(replace(job, data_path=cfg.data_path))
