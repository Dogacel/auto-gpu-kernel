"""Ninth probe: read blackwell.py and dot_fma source."""

import modal

app = modal.App("gluon-probe9")

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest", add_python="3.12")
    .apt_install("git", "wget", "build-essential", "cmake")
)


@app.function(image=image, gpu="B200:1", timeout=300)
def probe():
    import os

    # Read blackwell.py
    paths = [
        '/opt/conda/envs/py312/lib/python3.12/site-packages/triton/experimental/gluon/nvidia/blackwell.py',
        '/opt/conda/envs/py312/lib/python3.12/site-packages/triton/experimental/gluon/nvidia/hopper.py',
        '/opt/conda/envs/py312/lib/python3.12/site-packages/triton/experimental/gluon/language/nvidia/blackwell/__init__.py',
    ]
    for p in paths:
        print(f"\n===== {p} =====")
        with open(p) as f:
            print(f.read())


@app.local_entrypoint()
def main():
    probe.remote()
