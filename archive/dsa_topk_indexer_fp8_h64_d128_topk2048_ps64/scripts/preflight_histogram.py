"""Quick preflight: verify tl.histogram compiles and is correct on B200."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("preflight-histogram")

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest", add_python="3.12")
    .apt_install("git", "wget", "build-essential", "cmake")
)


@app.function(image=image, gpu="B200", timeout=300)
def test():
    import torch
    import triton
    import triton.language as tl

    print(f"triton version: {triton.__version__}")
    try:
        print(f"tl.histogram: {tl.histogram}")
    except AttributeError as e:
        print(f"NO tl.histogram: {e}")
        return

    N = 8192
    N_BUCKETS = 2048

    @triton.jit
    def hist_kernel(x_ptr, out_ptr, N: tl.constexpr, N_BUCKETS: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(x_ptr + offs)
        # Tests the 11-bit unsigned-range case
        h = tl.histogram(x, N_BUCKETS)
        tl.store(out_ptr + tl.arange(0, N_BUCKETS), h)

    # Test: random uint32 values shifted to have 11-bit bucket ids
    torch.manual_seed(0)
    x_ref = torch.randint(0, N_BUCKETS, (N,), device="cuda", dtype=torch.int32)
    out = torch.zeros(N_BUCKETS, device="cuda", dtype=torch.int32)

    hist_kernel[(1,)](x_ref, out, N=N, N_BUCKETS=N_BUCKETS)

    # Reference
    ref = torch.bincount(x_ref.to(torch.int64), minlength=N_BUCKETS).to(torch.int32)
    torch.cuda.synchronize()

    match = (out == ref).all().item()
    print(f"histogram match: {match}")
    if not match:
        diff = (out - ref).abs().sum().item()
        print(f"sum |diff|: {diff}")
        print(f"out[:20]={out[:20].tolist()}")
        print(f"ref[:20]={ref[:20].tolist()}")

    # Also test tl.flip + tl.cumsum for reverse-cumsum pattern we need
    @triton.jit
    def rev_cumsum_kernel(x_ptr, out_ptr, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(x_ptr + offs)
        xf = tl.flip(x, 0)
        cs = tl.cumsum(xf)
        rev_cs = tl.flip(cs, 0)
        tl.store(out_ptr + offs, rev_cs)

    x_test = torch.randint(0, 100, (N_BUCKETS,), device="cuda", dtype=torch.int32)
    out2 = torch.zeros_like(x_test)
    rev_cumsum_kernel[(1,)](x_test, out2, N=N_BUCKETS)
    ref2 = torch.flip(torch.cumsum(torch.flip(x_test, dims=[0]), dim=0), dims=[0]).to(torch.int32)
    torch.cuda.synchronize()
    print(f"rev_cumsum match: {(out2 == ref2).all().item()}")

    # Test argmax pattern: max bucket where rev_cs >= threshold
    @triton.jit
    def argmax_kernel(x_ptr, out_ptr, threshold, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(x_ptr + offs)
        mask = x >= threshold
        # max bucket index where mask
        idxs = tl.where(mask, offs, 0)
        result = tl.max(idxs)
        tl.store(out_ptr, result)

    rev = torch.flip(torch.cumsum(torch.flip(x_test, dims=[0]), dim=0), dims=[0]).to(torch.int32)
    out3 = torch.zeros(1, device="cuda", dtype=torch.int32)
    argmax_kernel[(1,)](rev, out3, 100, N=N_BUCKETS)
    ref3 = (rev >= 100).nonzero().max().item() if (rev >= 100).any() else 0
    torch.cuda.synchronize()
    print(f"argmax match: out={out3.item()} ref={ref3} {out3.item() == ref3}")


@app.local_entrypoint()
def run():
    test.remote()
