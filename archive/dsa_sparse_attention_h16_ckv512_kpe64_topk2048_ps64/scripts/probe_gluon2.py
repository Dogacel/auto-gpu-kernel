"""Second probe: test a minimal Gluon kernel."""

import modal

app = modal.App("gluon-probe2")

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

    # Probe more details
    import inspect
    for name in ['convert_layout', 'cast', 'reshape', 'expand_dims', 'broadcast',
                 'permute', 'barrier', 'log', 'log2', 'to_tensor']:
        if hasattr(gl, name):
            obj = getattr(gl, name)
            try:
                sig = str(inspect.signature(obj))
            except Exception:
                sig = "<no sig>"
            print(f"gl.{name}{sig}")
        else:
            print(f"gl.{name} NOT FOUND")

    # Try to compile a minimal Gluon kernel
    @gluon.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, N: gl.constexpr, BLOCK: gl.constexpr):
        layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1],
            threads_per_warp=[32],
            warps_per_cta=[4],
            order=[0],
        )
        pid = gl.program_id(0)
        offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
        mask = offs < N
        x = gl.load(x_ptr + offs, mask=mask)
        y = gl.load(y_ptr + offs, mask=mask)
        gl.store(out_ptr + offs, x + y, mask=mask)

    N = 1024
    x = torch.randn(N, device='cuda', dtype=torch.float32)
    y = torch.randn(N, device='cuda', dtype=torch.float32)
    out = torch.empty_like(x)

    BLOCK = 256
    grid = (triton.cdiv(N, BLOCK),)
    try:
        add_kernel[grid](x, y, out, N, BLOCK, num_warps=4)
        print("Gluon kernel compiled and ran!")
        print(f"match: {torch.allclose(out, x + y)}")
    except Exception as e:
        import traceback
        print(f"Gluon kernel FAILED: {e}")
        traceback.print_exc()

    # Try a matmul-like kernel using dot_fma
    print("\n=== Try dot_fma kernel ===")
    @gluon.jit
    def matmul_kernel(a_ptr, b_ptr, c_ptr,
                      M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                      BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr):
        layout_a: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 1],
            threads_per_warp=[8, 4],
            warps_per_cta=[2, 2],
            order=[1, 0],
        )
        layout_b: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 1],
            threads_per_warp=[8, 4],
            warps_per_cta=[2, 2],
            order=[1, 0],
        )
        layout_c: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 1],
            threads_per_warp=[8, 4],
            warps_per_cta=[2, 2],
            order=[1, 0],
        )
        offs_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, layout_c))
        offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, layout_c))
        offs_k_a = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, layout_a))
        offs_k_b = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, layout_b))

        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k_a[None, :]
        b_ptrs = b_ptr + offs_k_b[:, None] * N + offs_n[None, :]

        a = gl.load(a_ptrs)
        b = gl.load(b_ptrs)
        acc = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=layout_c)
        c = gl.dot_fma(a, b, acc)
        c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
        gl.store(c_ptrs, c)

    a = torch.randn(64, 64, device='cuda', dtype=torch.float32)
    b = torch.randn(64, 64, device='cuda', dtype=torch.float32)
    c = torch.empty(64, 64, device='cuda', dtype=torch.float32)
    try:
        matmul_kernel[(1,)](a, b, c, 64, 64, 64, 64, 64, 64, num_warps=4)
        c_ref = a @ b
        print(f"matmul match: {torch.allclose(c, c_ref, atol=1e-3, rtol=1e-3)}")
        print(f"max abs err: {(c - c_ref).abs().max().item()}")
    except Exception as e:
        import traceback
        print(f"matmul FAILED: {e}")
        traceback.print_exc()


@app.local_entrypoint()
def main():
    probe.remote()
