"""15th probe: softmax-like ops in Gluon."""

import modal

app = modal.App("gluon-probe15")

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest", add_python="3.12")
    .apt_install("git", "wget", "build-essential", "cmake")
)


@app.function(image=image, gpu="B200:1", timeout=300)
def probe():
    import torch
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    @gluon.jit
    def softmax_like(in_ptr, out_m_ptr, out_sum_ptr, H: gl.constexpr, N: gl.constexpr):
        layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 4],
            threads_per_warp=[4, 8],
            warps_per_cta=[1, 1],
            order=[1, 0],
        )
        offs_h = gl.arange(0, H, layout=gl.SliceLayout(1, layout))[:, None]
        offs_n = gl.arange(0, N, layout=gl.SliceLayout(0, layout))[None, :]
        x = gl.load(in_ptr + offs_h * N + offs_n)
        # max along axis=1
        m = gl.max(x, axis=1)
        # x - m (broadcasting)
        m_exp = m[:, None]
        # convert to match x layout:
        diff = x - m_exp
        p = gl.exp2(diff)
        s = gl.sum(p, axis=1)

        # Need to store m and s (shape [H])
        offs_h_1d = gl.arange(0, H, layout=gl.SliceLayout(1, layout))
        gl.store(out_m_ptr + offs_h_1d, m)
        gl.store(out_sum_ptr + offs_h_1d, s)

    H, N = 16, 64
    x = torch.randn(H, N, device='cuda', dtype=torch.float32)
    m = torch.empty(H, device='cuda', dtype=torch.float32)
    s = torch.empty(H, device='cuda', dtype=torch.float32)
    try:
        softmax_like[(1,)](x, m, s, H, N, num_warps=1)
        torch.cuda.synchronize()
        m_ref = x.max(dim=1).values
        s_ref = ((x - m_ref[:, None]) * (1 / torch.log(torch.tensor(2.0)))).exp().sum(dim=1)
        # using exp2: p = 2^(x-m) = exp((x-m)*ln2) = exp(x-m)*... no, exp2(x-m) directly
        s_ref2 = (x - m_ref[:, None]).exp2().sum(dim=1)
        print(f"m match: {torch.allclose(m, m_ref)}")
        print(f"s match: {torch.allclose(s, s_ref2, atol=1e-4)}")
        print(f"max err m: {(m - m_ref).abs().max().item()}")
        print(f"max err s: {(s - s_ref2).abs().max().item()}")
    except Exception as e:
        import traceback
        traceback.print_exc()


@app.local_entrypoint()
def main():
    probe.remote()
