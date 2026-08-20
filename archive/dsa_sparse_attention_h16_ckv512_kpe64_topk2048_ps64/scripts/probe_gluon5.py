"""Fifth probe: dot_fma correctness + atomic + MMA with shared memory."""

import modal

app = modal.App("gluon-probe5")

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
    from triton.experimental.gluon.language.nvidia import blackwell as bw

    # Try dot_fma with DotOperandLayout (no k_width) on BlockedLayout parent - fp32
    @gluon.jit
    def matmul_fma(a_ptr, b_ptr, c_ptr,
                   M: gl.constexpr, N: gl.constexpr, K: gl.constexpr):
        out_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 4],
            threads_per_warp=[8, 4],
            warps_per_cta=[2, 2],
            order=[1, 0],
        )
        dot_a_layout: gl.constexpr = gl.DotOperandLayout(
            operand_index=0,
            parent=out_layout,
        )
        dot_b_layout: gl.constexpr = gl.DotOperandLayout(
            operand_index=1,
            parent=out_layout,
        )
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

    M, N, K = 16, 64, 64
    a = torch.randn(M, K, device='cuda', dtype=torch.float32)
    b = torch.randn(K, N, device='cuda', dtype=torch.float32)
    c = torch.empty(M, N, device='cuda', dtype=torch.float32)
    try:
        matmul_fma[(1,)](a, b, c, M, N, K, num_warps=4)
        c_ref = a @ b
        print(f"fp32 dot_fma match: {torch.allclose(c, c_ref, atol=1e-2, rtol=1e-2)}")
        print(f"fp32 max abs err: {(c - c_ref).abs().max().item()}")
    except Exception as e:
        import traceback
        print(f"FAILED: {e}")
        traceback.print_exc()

    # bf16 test
    @gluon.jit
    def matmul_bf16(a_ptr, b_ptr, c_ptr,
                    M: gl.constexpr, N: gl.constexpr, K: gl.constexpr):
        out_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 4],
            threads_per_warp=[8, 4],
            warps_per_cta=[2, 2],
            order=[1, 0],
        )
        dot_a_layout: gl.constexpr = gl.DotOperandLayout(
            operand_index=0,
            parent=out_layout,
        )
        dot_b_layout: gl.constexpr = gl.DotOperandLayout(
            operand_index=1,
            parent=out_layout,
        )
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
        gl.store(c_ptr + offs_m_out * N + offs_n_out, c.to(gl.bfloat16))

    M, N, K = 16, 64, 64
    a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)
    c = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)
    try:
        matmul_bf16[(1,)](a, b, c, M, N, K, num_warps=4)
        c_ref = (a.float() @ b.float()).to(torch.bfloat16)
        print(f"\nbf16 dot_fma match: {torch.allclose(c, c_ref, atol=1e-1, rtol=1e-1)}")
        print(f"bf16 max abs err: {(c.float() - c_ref.float()).abs().max().item()}")
    except Exception as e:
        import traceback
        print(f"FAILED: {e}")
        traceback.print_exc()

    # Try larger matmul 16x128 @ 128x32 (like ours with H=16, D=64, BLOCK_N=128)
    print("\n=== test 16 x 128 x 64 (like our Q@Kpe) ===")
    @gluon.jit
    def matmul_small(a_ptr, b_ptr, c_ptr,
                     M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                     num_warps_x_4: gl.constexpr):
        # num_warps=8
        out_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 8],
            threads_per_warp=[4, 8],
            warps_per_cta=[num_warps_x_4, 1],
            order=[1, 0],
        )
        dot_a_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=out_layout)
        dot_b_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=out_layout)
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
        gl.store(c_ptr + offs_m_out * N + offs_n_out, c.to(gl.bfloat16))

    M, N, K = 16, 128, 64
    a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)
    c = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)
    try:
        matmul_small[(1,)](a, b, c, M, N, K, 2, num_warps=8)
        c_ref = (a.float() @ b.float()).to(torch.bfloat16)
        print(f"Qpe@K shape match: {torch.allclose(c, c_ref, atol=1e-1, rtol=1e-1)}")
        print(f"max abs err: {(c.float() - c_ref.float()).abs().max().item()}")
    except Exception as e:
        import traceback
        print(f"FAILED: {e}")
        traceback.print_exc()


@app.local_entrypoint()
def main():
    probe.remote()
