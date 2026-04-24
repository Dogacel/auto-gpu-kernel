"""Seventh probe: what k_width value works?"""

import modal

app = modal.App("gluon-probe7")

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest", add_python="3.12")
    .apt_install("git", "wget", "build-essential", "cmake")
)


@app.function(image=image, gpu="B200:1", timeout=300)
def probe():
    import torch
    import triton
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    import inspect
    from triton.experimental.gluon.language import _layouts
    print(inspect.getsource(_layouts.DotOperandLayout))


@app.local_entrypoint()
def main():
    probe.remote()
