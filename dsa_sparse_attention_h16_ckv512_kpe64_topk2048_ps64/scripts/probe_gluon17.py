"""17th probe: test 16x512x128 dot_fma correctness in isolation."""

import modal

app = modal.App("gluon-probe17")

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest", add_python="3.12")
    .apt_install("git", "wget", "build-essential", "cmake")
)


@app.function(image=image, gpu="B200:1", timeout=300)
def probe():
    import torch
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    # Test: q_nope [16, 512] @ kc^T [512, 128] -> [16, 128], bf16 in, fp32 out
    @gluon.jit
    def qkt(qn_ptr, kc_ptr, out_ptr,
            H: gl.constexpr, D: gl.constexpr, N: gl.constexpr):
        # Layouts
        pn_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 4],
            threads_per_warp=[1, 32],
            warps_per_cta=[8, 1],
            order=[1, 0],
        )
        qn_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 8],
            threads_per_warp=[1, 32],
            warps_per_cta=[8, 1],
            order=[1, 0],
        )
        kc_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 8],
            threads_per_warp=[16, 2],
            warps_per_cta=[8, 1],
            order=[1, 0],
        )
        # Load q_nope
        offs_h = gl.arange(0, H, layout=gl.SliceLayout(1, qn_layout))[:, None]
        offs_d = gl.arange(0, D, layout=gl.SliceLayout(0, qn_layout))[None, :]
        qn = gl.load(qn_ptr + offs_h * D + offs_d)

        # Load kc
        offs_n = gl.arange(0, N, layout=gl.SliceLayout(1, kc_layout))[:, None]
        offs_d_kc = gl.arange(0, D, layout=gl.SliceLayout(0, kc_layout))[None, :]
        kc = gl.load(kc_ptr + offs_n * D + offs_d_kc)

        # Transpose kc [N, D] -> [D, N]
        kc_t = gl.permute(kc, (1, 0))  # [D, N]

        # Cast to fp32 and convert to DotOperand layouts
        qn_f32 = qn.to(gl.float32)
        kc_t_f32 = kc_t.to(gl.float32)
        dot_q: gl.constexpr = gl.DotOperandLayout(0, pn_layout, 0)
        dot_k: gl.constexpr = gl.DotOperandLayout(1, pn_layout, 0)
        qn_d = gl.convert_layout(qn_f32, dot_q)
        kc_t_d = gl.convert_layout(kc_t_f32, dot_k)
        acc = gl.zeros([H, N], dtype=gl.float32, layout=pn_layout)
        out = gl.dot_fma(qn_d, kc_t_d, acc)

        # Store
        offs_h2 = gl.arange(0, H, layout=gl.SliceLayout(1, pn_layout))[:, None]
        offs_n2 = gl.arange(0, N, layout=gl.SliceLayout(0, pn_layout))[None, :]
        gl.store(out_ptr + offs_h2 * N + offs_n2, out)

    H, D, N = 16, 512, 128
    qn = torch.randn(H, D, device='cuda', dtype=torch.bfloat16)
    kc = torch.randn(N, D, device='cuda', dtype=torch.bfloat16)
    out = torch.empty(H, N, device='cuda', dtype=torch.float32)
    try:
        qkt[(1,)](qn, kc, out, H, D, N, num_warps=8)
        torch.cuda.synchronize()
        out_ref = qn.float() @ kc.float().T
        abs_err = (out - out_ref).abs().max().item()
        print(f"QKT: match={torch.allclose(out, out_ref, atol=1e-2, rtol=1e-2)}, abs_err={abs_err:.4e}")
    except Exception as e:
        import traceback
        traceback.print_exc()


@app.local_entrypoint()
def main():
    probe.remote()
