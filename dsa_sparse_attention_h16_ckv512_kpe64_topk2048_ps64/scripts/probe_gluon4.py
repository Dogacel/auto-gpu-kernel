"""Fourth probe: explore dot_fma with DotOperandLayout over BlockedLayout parent."""

import modal

app = modal.App("gluon-probe4")

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

    # Try dot_fma with DotOperandLayout on BlockedLayout parent
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
            k_width=1,
        )
        dot_b_layout: gl.constexpr = gl.DotOperandLayout(
            operand_index=1,
            parent=out_layout,
            k_width=1,
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
        print(f"dot_fma match: {torch.allclose(c, c_ref, atol=1e-2, rtol=1e-2)}")
        print(f"max abs err: {(c - c_ref).abs().max().item()}")
    except Exception as e:
        import traceback
        print(f"FAILED: {e}")
        traceback.print_exc()

    # Check bf16 as well since that's our real use case
    print("\n=== bf16 dot_fma with accumulation ===")
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
            k_width=1,
        )
        dot_b_layout: gl.constexpr = gl.DotOperandLayout(
            operand_index=1,
            parent=out_layout,
            k_width=1,
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
        print(f"bf16 dot_fma match: {torch.allclose(c, c_ref, atol=1e-1, rtol=1e-1)}")
        print(f"max abs err: {(c.float() - c_ref.float()).abs().max().item()}")
    except Exception as e:
        import traceback
        print(f"FAILED: {e}")
        traceback.print_exc()

    # Test atomic_add and a while-loop
    print("\n=== atomic_add + spin-wait ===")
    @gluon.jit
    def atomic_spin(ctr_ptr, out_ptr, NUM: gl.constexpr):
        layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1],
            threads_per_warp=[32],
            warps_per_cta=[1],
            order=[0],
        )
        pid = gl.program_id(0)
        # Each CTA increments counter
        gl.atomic_add(ctr_ptr, 1, sem="release")
        # Spin
        count = gl.load(ctr_ptr, volatile=True)
        while count < NUM:
            count = gl.load(ctr_ptr, volatile=True)
        # Write our pid
        idx = gl.arange(0, 1, layout=layout)
        gl.store(out_ptr + pid + idx, pid + idx)

    N = 8
    ctr = torch.zeros(1, dtype=torch.int32, device='cuda')
    out = torch.zeros(N, dtype=torch.int32, device='cuda')
    try:
        atomic_spin[(N,)](ctr, out, N, num_warps=1)
        torch.cuda.synchronize()
        print(f"atomic+spin ok: ctr={ctr.item()}, out={out.tolist()}")
    except Exception as e:
        import traceback
        print(f"FAILED: {e}")
        traceback.print_exc()


@app.local_entrypoint()
def main():
    probe.remote()
