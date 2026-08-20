"""18th probe: test p @ kc with layout conversion."""

import modal

app = modal.App("gluon-probe18")

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest", add_python="3.12")
    .apt_install("git", "wget", "build-essential", "cmake")
)


@app.function(image=image, gpu="B200:1", timeout=300)
def probe():
    import torch
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    # Test: p [H, N] @ kc [N, D] -> [H, D]
    # p's layout should be pn_layout; convert to DotOperand(0, qn_layout, 0)
    # kc's layout kc_layout; convert to DotOperand(1, qn_layout, 0)
    @gluon.jit
    def p_mm_kc(p_ptr, kc_ptr, out_ptr,
                H: gl.constexpr, N: gl.constexpr, D: gl.constexpr):
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
        # Load p
        offs_h = gl.arange(0, H, layout=gl.SliceLayout(1, pn_layout))[:, None]
        offs_n = gl.arange(0, N, layout=gl.SliceLayout(0, pn_layout))[None, :]
        p = gl.load(p_ptr + offs_h * N + offs_n)

        # Load kc
        offs_n_kc = gl.arange(0, N, layout=gl.SliceLayout(1, kc_layout))[:, None]
        offs_d_kc = gl.arange(0, D, layout=gl.SliceLayout(0, kc_layout))[None, :]
        kc = gl.load(kc_ptr + offs_n_kc * D + offs_d_kc)

        # Convert to DotOperand (parent=qn_layout) and compute
        dot_p: gl.constexpr = gl.DotOperandLayout(0, qn_layout, 0)
        dot_k: gl.constexpr = gl.DotOperandLayout(1, qn_layout, 0)
        p_d = gl.convert_layout(p, dot_p)
        kc_d = gl.convert_layout(kc, dot_k)
        acc = gl.zeros([H, D], dtype=gl.float32, layout=qn_layout)
        out = gl.dot_fma(p_d, kc_d, acc)

        # Store (convert to qn_layout for store)
        offs_h2 = gl.arange(0, H, layout=gl.SliceLayout(1, qn_layout))[:, None]
        offs_d2 = gl.arange(0, D, layout=gl.SliceLayout(0, qn_layout))[None, :]
        gl.store(out_ptr + offs_h2 * D + offs_d2, out)

    H, N, D = 16, 128, 512
    p = torch.randn(H, N, device='cuda', dtype=torch.float32)
    kc = torch.randn(N, D, device='cuda', dtype=torch.float32)
    out = torch.empty(H, D, device='cuda', dtype=torch.float32)
    try:
        p_mm_kc[(1,)](p, kc, out, H, N, D, num_warps=8)
        torch.cuda.synchronize()
        out_ref = p @ kc
        abs_err = (out - out_ref).abs().max().item()
        print(f"p@kc: match={torch.allclose(out, out_ref, atol=1e-2, rtol=1e-2)}, abs_err={abs_err:.4e}")
        print(f"out[0, :8] = {out[0, :8].tolist()}")
        print(f"ref[0, :8] = {out_ref[0, :8].tolist()}")
    except Exception as e:
        import traceback
        traceback.print_exc()


@app.local_entrypoint()
def main():
    probe.remote()
