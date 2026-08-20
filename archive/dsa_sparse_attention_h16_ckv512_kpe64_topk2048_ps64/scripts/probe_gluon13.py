"""13th probe: permute/transpose, mask, dot acc=, volatile load."""

import modal

app = modal.App("gluon-probe13")

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

    # Does dot_fma support acc= parameter (or is acc always the third arg, additive)?
    # From the source, dot_fma(a, b, acc) returns acc + a @ b

    # Test: transpose via permute
    @gluon.jit
    def transpose_test(in_ptr, out_ptr, M: gl.constexpr, N: gl.constexpr):
        layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 1],
            threads_per_warp=[8, 4],
            warps_per_cta=[1, 1],
            order=[1, 0],
        )
        offs_m = gl.arange(0, M, layout=gl.SliceLayout(1, layout))[:, None]
        offs_n = gl.arange(0, N, layout=gl.SliceLayout(0, layout))[None, :]
        x = gl.load(in_ptr + offs_m * N + offs_n)
        xt = gl.permute(x, (1, 0))

        # Now store xt which is [N, M]
        layout_t: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 1],
            threads_per_warp=[4, 8],
            warps_per_cta=[1, 1],
            order=[1, 0],
        )
        xt_conv = gl.convert_layout(xt, layout_t)
        offs_n_out = gl.arange(0, N, layout=gl.SliceLayout(1, layout_t))[:, None]
        offs_m_out = gl.arange(0, M, layout=gl.SliceLayout(0, layout_t))[None, :]
        gl.store(out_ptr + offs_n_out * M + offs_m_out, xt_conv)

    M, N = 8, 32
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    y = torch.empty(N, M, device='cuda', dtype=torch.float32)
    try:
        transpose_test[(1,)](x, y, M, N, num_warps=1)
        torch.cuda.synchronize()
        print(f"transpose match: {torch.allclose(y, x.T)}")
    except Exception as e:
        import traceback
        print(f"transpose FAILED: {str(e)[:300]}")

    # Masked load
    print("\n=== masked load ===")
    @gluon.jit
    def masked_load_test(in_ptr, out_ptr, N: gl.constexpr):
        layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[4],
            threads_per_warp=[32],
            warps_per_cta=[1],
            order=[0],
        )
        offs = gl.arange(0, N, layout=layout)
        mask = offs < (N // 2)
        x = gl.load(in_ptr + offs, mask=mask, other=-1.0)
        gl.store(out_ptr + offs, x)

    N = 128
    x = torch.ones(N, device='cuda', dtype=torch.float32)
    y = torch.empty_like(x)
    try:
        masked_load_test[(1,)](x, y, N, num_warps=1)
        torch.cuda.synchronize()
        expected = torch.cat([torch.ones(N//2), -torch.ones(N//2)]).cuda()
        print(f"masked match: {torch.allclose(y, expected)}")
    except Exception as e:
        import traceback
        print(f"masked FAILED: {str(e)[:300]}")

    # Volatile load
    print("\n=== volatile load ===")
    @gluon.jit
    def vol_test(ctr_ptr, out_ptr, N: gl.constexpr):
        layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1],
            threads_per_warp=[32],
            warps_per_cta=[1],
            order=[0],
        )
        pid = gl.program_id(0)
        if pid == 0:
            # Write
            gl.store(ctr_ptr, 42)
        # Read with volatile
        # Reading scalars through pointers: need to make it a tensor?
        # gl.load(ctr_ptr) on a pointer (non-tensor) -> scalar load
        x = gl.load(ctr_ptr, volatile=True)
        gl.store(out_ptr + pid + gl.arange(0, 1, layout=layout), x + gl.arange(0, 1, layout=layout))

    N = 1
    ctr = torch.zeros(1, dtype=torch.int32, device='cuda')
    out = torch.zeros(4, dtype=torch.int32, device='cuda')
    try:
        vol_test[(4,)](ctr, out, N, num_warps=1)
        torch.cuda.synchronize()
        print(f"volatile load result: ctr={ctr.item()}, out={out.tolist()}")
    except Exception as e:
        import traceback
        print(f"volatile FAILED: {str(e)[:300]}")
        traceback.print_exc()


@app.local_entrypoint()
def main():
    probe.remote()
