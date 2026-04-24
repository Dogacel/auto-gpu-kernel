"""Tenth probe: dot_fma source, look at _core.py."""

import modal

app = modal.App("gluon-probe10")

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest", add_python="3.12")
    .apt_install("git", "wget", "build-essential", "cmake")
)


@app.function(image=image, gpu="B200:1", timeout=300)
def probe():
    import inspect
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language import _core, _semantic, _layouts

    print("=== dot_fma source ===")
    print(inspect.getsource(_core.dot_fma))

    print("\n=== _semantic.dot_fma (if exists) ===")
    if hasattr(_semantic, 'GluonSemantic'):
        sem = _semantic.GluonSemantic
        if hasattr(sem, 'dot_fma'):
            print(inspect.getsource(sem.dot_fma))

    print("\n=== DotOperandLayout class ===")
    print(inspect.getsource(_layouts.DotOperandLayout))

    # look at tests for Gluon kernels? Also look at _semantic for load signature
    print("\n=== ttgl namespace ===")
    from triton.experimental.gluon.language import nvidia
    for k in dir(nvidia):
        print(f"  {k}")


@app.local_entrypoint()
def main():
    probe.remote()
