"""Probe Gluon API on Modal to discover available primitives."""

import modal

app = modal.App("gluon-probe")

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest", add_python="3.12")
    .apt_install("git", "wget", "build-essential", "cmake")
)


@app.function(image=image, gpu="B200:1", timeout=300)
def probe():
    import triton
    print(f"Triton version: {triton.__version__}")

    # Core Gluon imports
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    print("\n=== gluon module ===")
    print([x for x in dir(gluon) if not x.startswith('_')])

    print("\n=== gluon.language module ===")
    print([x for x in dir(gl) if not x.startswith('_')])

    # Blackwell-specific
    try:
        from triton.experimental.gluon.language.nvidia import blackwell as bw
        print("\n=== gluon.language.nvidia.blackwell ===")
        print([x for x in dir(bw) if not x.startswith('_')])

        print("\n=== bw.mbarrier ===")
        print([x for x in dir(bw.mbarrier) if not x.startswith('_')])

        print("\n=== bw.tma ===")
        print([x for x in dir(bw.tma) if not x.startswith('_')])

        print("\n=== bw.tcgen05_mma ===")
        print([x for x in dir(bw.tcgen05_mma) if not x.startswith('_')])

        print("\n=== bw.async_copy ===")
        print([x for x in dir(bw.async_copy) if not x.startswith('_')])
    except ImportError as e:
        print(f"blackwell import failed: {e}")

    # Layout objects
    print("\n=== gl.BlockedLayout ===")
    try:
        help(gl.BlockedLayout)
    except Exception as e:
        print(f"err: {e}")

    # Key functions
    for name in ['load', 'store', 'atomic_add', 'thread_barrier', 'dot_fma',
                 'allocate_shared_memory', 'warp_specialize',
                 'inline_asm_elementwise', 'NVMMADistributedLayout',
                 'arange', 'zeros', 'full', 'maximum', 'exp2', 'sum', 'max',
                 'where', 'to']:
        if hasattr(gl, name):
            obj = getattr(gl, name)
            sig = ""
            try:
                import inspect
                sig = str(inspect.signature(obj))
            except Exception:
                sig = "<no sig>"
            print(f"gl.{name}{sig}")
        else:
            print(f"gl.{name} NOT FOUND")


@app.local_entrypoint()
def main():
    probe.remote()
