"""Third probe: explore DotOperandLayout, NVMMADistributedLayout, matmul usage."""

import modal

app = modal.App("gluon-probe3")

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

    import inspect

    # Check DotOperandLayout
    for name in ['DotOperandLayout', 'SliceLayout', 'NVMMADistributedLayout',
                 'NVMMASharedLayout', 'CoalescedLayout']:
        try:
            obj = getattr(gl, name)
            print(f"\n=== {name} ===")
            try:
                sig = str(inspect.signature(obj.__init__))
                print(sig)
            except Exception:
                pass
            # Print any custom methods
            methods = [x for x in dir(obj) if not x.startswith('_')]
            print(f"methods: {methods}")
        except Exception as e:
            print(f"{name} err: {e}")

    # Minimal matmul with NVMMA
    print("\n=== matmul with NVMMA ===")

    @gluon.jit
    def matmul_nvmma(a_ptr, b_ptr, c_ptr,
                     M: gl.constexpr, N: gl.constexpr, K: gl.constexpr):
        # NVMMA layout for MMA output (warps_per_cta [warps_m, warps_n])
        mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
            version=[3, 0],
            warps_per_cta=[2, 2],
            instr_shape=[16, 16, 16],
        )
        # Blocked layouts for loads (then convert)
        blocked: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 4],
            threads_per_warp=[8, 4],
            warps_per_cta=[2, 2],
            order=[1, 0],
        )
        offs_m = gl.arange(0, M, layout=gl.SliceLayout(1, blocked))[:, None]
        offs_n = gl.arange(0, N, layout=gl.SliceLayout(0, blocked))[None, :]
        offs_k_col = gl.arange(0, K, layout=gl.SliceLayout(0, blocked))[None, :]
        offs_k_row = gl.arange(0, K, layout=gl.SliceLayout(1, blocked))[:, None]

        a = gl.load(a_ptr + offs_m * K + offs_k_col)
        b = gl.load(b_ptr + offs_k_row * N + offs_n)

        # Convert to DotOperand layouts
        dot_a_layout: gl.constexpr = gl.DotOperandLayout(
            operand_index=0,
            parent=mma_layout,
            k_width=2,
        )
        dot_b_layout: gl.constexpr = gl.DotOperandLayout(
            operand_index=1,
            parent=mma_layout,
            k_width=2,
        )
        a_dot = gl.convert_layout(a, dot_a_layout)
        b_dot = gl.convert_layout(b, dot_b_layout)
        acc = gl.zeros([M, N], dtype=gl.float32, layout=mma_layout)
        c = gl.dot_fma(a_dot, b_dot, acc)

        c_ptrs = c_ptr + offs_m * N + offs_n
        c_converted = gl.convert_layout(c, blocked)
        gl.store(c_ptrs, c_converted)

    M, N, K = 64, 64, 64
    a = torch.randn(M, K, device='cuda', dtype=torch.float32)
    b = torch.randn(K, N, device='cuda', dtype=torch.float32)
    c = torch.empty(M, N, device='cuda', dtype=torch.float32)
    try:
        matmul_nvmma[(1,)](a, b, c, M, N, K, num_warps=4)
        c_ref = a @ b
        print(f"match: {torch.allclose(c, c_ref, atol=1e-2, rtol=1e-2)}")
        print(f"max abs err: {(c - c_ref).abs().max().item()}")
    except Exception as e:
        import traceback
        print(f"FAILED: {e}")
        traceback.print_exc()


@app.local_entrypoint()
def main():
    probe.remote()
