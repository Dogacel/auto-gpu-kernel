"""Eighth probe: find Gluon example kernels."""

import modal

app = modal.App("gluon-probe8")

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest", add_python="3.12")
    .apt_install("git", "wget", "build-essential", "cmake")
)


@app.function(image=image, gpu="B200:1", timeout=300)
def probe():
    import os
    import triton
    # Find triton package path
    triton_path = os.path.dirname(triton.__file__)
    print(f"Triton path: {triton_path}")

    # Look for gluon examples / tests
    for root, dirs, files in os.walk(triton_path):
        for f in files:
            if 'gluon' in f.lower() and f.endswith('.py'):
                full = os.path.join(root, f)
                # skip __pycache__
                if '__pycache__' in full:
                    continue
                print(full)

    # Check test dir
    import site
    for sp in site.getsitepackages():
        tests_dir = os.path.join(sp, 'triton', 'experimental', 'gluon', 'language', 'nvidia', 'hopper')
        if os.path.exists(tests_dir):
            print(f"\nHopper files: {os.listdir(tests_dir)}")

    # Find anything with 'attention' or 'flash' in experimental
    for root, dirs, files in os.walk(os.path.join(triton_path, 'experimental')):
        for f in files:
            if f.endswith('.py') and '__pycache__' not in root:
                full = os.path.join(root, f)
                print(full)


@app.local_entrypoint()
def main():
    probe.remote()
