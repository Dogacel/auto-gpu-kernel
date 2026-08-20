"""12th probe: hoist DotOperandLayout out of constexpr and literal k_width=0."""

import modal

app = modal.App("gluon-probe12")

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

    # Try with literal k_width=0
    @gluon.jit
    def matmul_fma_kw0(a_ptr, b_ptr, c_ptr,
                   M: gl.constexpr, N: gl.constexpr, K: gl.constexpr):
        out_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 4],
            threads_per_warp=[8, 4],
            warps_per_cta=[2, 2],
            order=[1, 0],
        )
        dot_a_layout: gl.constexpr = gl.DotOperandLayout(0, out_layout, 0)
        dot_b_layout: gl.constexpr = gl.DotOperandLayout(1, out_layout, 0)
        offs_m_a = gl.arange(0, M, layout=gl.SliceLayout(1, dot_a_layout))[:, None]
        offs_k_a = gl.arange(0, K, layout=gl.SliceLayout(0, dot_a_layout))[None, :]
        offs_k_b = gl.arange(0, K, layout=gl.SliceLayout(1, dot_b_layout))[:, None]
        offs_n_b = gl.arange(0, N, layout=gl.SliceLayout(0, dot_b_layout))[None, :]

        a = gl.load(a_ptr + offs_m_a * K + offs_k_a)
        b = gl.load(b_ptr + offs_k_b * N + offs_n_b)

        acc = gl.zeros([M, N], dtype=gl.float32, layout=out_layout)
        c = gl.dot_fma(a, b, acc)

        offs_m_out = gl.arange(0, M, layout=gl.SliceLayout(1, out_layout))[:, None]
        offs_n_out = gl.arange(0, N, layout=gl.SliceLayout(0, out_layout))[None, :]
        gl.store(c_ptr + offs_m_out * N + offs_n_out, c)

    print("=== dot_fma with k_width=0 literal ===")
    M, N, K = 16, 64, 64
    a = torch.randn(M, K, device='cuda', dtype=torch.float32)
    b = torch.randn(K, N, device='cuda', dtype=torch.float32)
    c = torch.empty(M, N, device='cuda', dtype=torch.float32)
    try:
        matmul_fma_kw0[(1,)](a, b, c, M, N, K, num_warps=4)
        torch.cuda.synchronize()
        c_ref = a @ b
        print(f"  match: {torch.allclose(c, c_ref, atol=1e-2, rtol=1e-2)}")
        print(f"  max err: {(c - c_ref).abs().max().item()}")
    except Exception as e:
        import traceback
        traceback.print_exc()


@app.local_entrypoint()
def main():
    probe.remote()
