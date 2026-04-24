"""16th probe: gl.barrier availability."""

import modal

app = modal.App("gluon-probe16")

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest", add_python="3.12")
    .apt_install("git", "wget", "build-essential", "cmake")
)


@app.function(image=image, gpu="B200:1", timeout=300)
def probe():
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language import _core
    # List everything
    print("gl.barrier:", getattr(gl, 'barrier', None))
    print("has barrier:", hasattr(gl, 'barrier'))
    print("_core.barrier:", getattr(_core, 'barrier', None))

    # Check all in __init__.py
    import os
    path = '/opt/conda/envs/py312/lib/python3.12/site-packages/triton/experimental/gluon/language/__init__.py'
    with open(path) as f:
        content = f.read()
    print("\n=== __init__.py ===")
    print(content)


@app.local_entrypoint()
def main():
    probe.remote()
