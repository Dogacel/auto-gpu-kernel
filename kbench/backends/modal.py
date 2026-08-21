"""Modal backend. Image is replayed from the config's ImageSpec."""

from __future__ import annotations

from dataclasses import replace

from kbench.job import Job
from kbench.results import WorkloadResult


def _image(cfg):
    import modal

    img = modal.Image.from_registry(cfg.image.base, add_python="3.12")
    if cfg.image.apt:
        img = img.apt_install(*cfg.image.apt)
    if cfg.image.pip:
        img = img.pip_install(*cfg.image.pip)
    for cmd in cfg.image.run:
        img = img.run_commands(cmd)
    if cfg.image.env:
        img = img.env(dict(cfg.image.env))
    # Ships the kbench package so flashinfer.evaluate resolves remotely.
    return img.add_local_python_source("kbench")


def run(cfg, job: Job) -> dict[str, list[WorkloadResult]]:
    import modal

    from kbench.flashinfer import evaluate

    app = modal.App("kbench")
    volume = modal.Volume.from_name(cfg.modal_volume, create_if_missing=True)

    remote = app.function(
        image=_image(cfg),
        gpu=f"{cfg.gpu}:{cfg.gpu_count}",
        timeout=cfg.timeout_s,
        volumes={cfg.data_path: volume},
    )(evaluate)

    with modal.enable_output(), app.run():
        return remote.remote(replace(job, data_path=cfg.data_path))
