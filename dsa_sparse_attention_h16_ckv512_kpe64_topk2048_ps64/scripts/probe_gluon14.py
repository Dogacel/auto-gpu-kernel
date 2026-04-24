"""14th probe: test softmax ops (max, sum axis=1), reduce, int32 load."""

import modal

app = modal.App("gluon-probe14")

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest", add_python="3.12")
    .apt_install("git", "wget", "build-essential", "cmake")
)


@app.function(image=image, gpu="B200:1", timeout=300)
def probe():
    import torch
    import inspect
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language._core import max, sum, maximum

    # Check signatures for max/sum
    print("=== gl.max signature ===")
    try:
        print(inspect.getsource(max))
    except Exception as e:
        print(e)

    print("\n=== gl.sum signature ===")
    try:
        print(inspect.getsource(sum))
    except Exception as e:
        print(e)


@app.local_entrypoint()
def main():
    probe.remote()
