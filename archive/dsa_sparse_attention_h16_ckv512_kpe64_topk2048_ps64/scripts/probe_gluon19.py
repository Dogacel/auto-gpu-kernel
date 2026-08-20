"""19th probe: verify TMEM .slice returns only top H rows after MMA."""

import modal

app = modal.App("gluon-probe19")

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest", add_python="3.12")
    .apt_install("git", "wget", "build-essential", "cmake")
)


@app.function(image=image, gpu="B200:1", timeout=600)
def probe():
    import torch
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.nvidia.hopper import mbarrier, fence_async_shared
    from triton.experimental.gluon.language.nvidia.blackwell import (
        TensorMemoryLayout,
        allocate_tensor_memory,
        get_tmem_reg_layout,
        tcgen05_mma,
        tcgen05_commit,
    )

    @gluon.jit
    def toy(Q_ptr, K_ptr, Out_ptr,
            H: gl.constexpr, HM: gl.constexpr, N: gl.constexpr, D: gl.constexpr):
        q_blk: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0])
        k_blk: gl.constexpr = gl.BlockedLayout([1, 8], [8, 4], [4, 1], [1, 0])

        oh = gl.arange(0, HM, layout=gl.SliceLayout(1, q_blk))[:, None]
        od = gl.arange(0, D, layout=gl.SliceLayout(0, q_blk))[None, :]
        q = gl.load(Q_ptr + oh * D + od, mask=oh < H, other=0.0)
        onk = gl.arange(0, N, layout=gl.SliceLayout(1, k_blk))[:, None]
        odk = gl.arange(0, D, layout=gl.SliceLayout(0, k_blk))[None, :]
        k = gl.load(K_ptr + onk * D + odk)
        k_t = gl.permute(k, (1, 0))

        qa: gl.constexpr = gl.NVMMASharedLayout(128, 16, 2, False)
        kb: gl.constexpr = gl.NVMMASharedLayout(128, 16, 2, True)
        q_smem = gl.allocate_shared_memory(gl.bfloat16, [HM, D], qa, value=q)
        k_smem = gl.allocate_shared_memory(gl.bfloat16, [D, N], kb, value=k_t)

        tmem_layout: gl.constexpr = TensorMemoryLayout([HM, N], col_stride=1)
        tmem_reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (HM, N), tmem_layout, 4)
        acc0 = gl.zeros([HM, N], gl.float32, layout=tmem_reg_layout)
        tmem_acc = allocate_tensor_memory(gl.float32, [HM, N], tmem_layout, acc0)
        fence_async_shared()

        bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(bar, count=1)
        tcgen05_mma(q_smem, k_smem, tmem_acc, use_acc=True)
        tcgen05_commit(bar)
        mbarrier.wait(bar, phase=0)
        mbarrier.invalidate(bar)

        # Try to slice TMEM to get only [H, N].
        tmem_h = tmem_acc.slice(0, H)  # slice along last dim? Let's see
        # Or is slice along dim=0? Let's check:
        # slice(self, start, length, _semantic=None) -> 'None'  — docs say "along last dimension"
        # That means this slices N, not the rows!
        # Instead: just load the full [HM, N] and use register slicing after layout-convert.

        logits_big = tmem_acc.load(tmem_reg_layout)
        # Try to reshape/convert to a smaller tile?
        # BlockedLayout output layout for [H, N]:
        out_blk: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])

        # Convert to pn_layout for broadcasting compatibility.
        # Layout conversion preserves shape; we can't shrink via convert_layout.
        # Approach: pick a TMEM descriptor of shape [H, N] whose block maps to the
        # top part of the big block. Use alloc_shape in descriptor.

        # Alternative: return logits_big, store only top H rows.
        oho = gl.arange(0, HM, layout=gl.SliceLayout(1, tmem_reg_layout))[:, None]
        ono = gl.arange(0, N, layout=gl.SliceLayout(0, tmem_reg_layout))[None, :]
        gl.store(Out_ptr + oho * N + ono, logits_big, mask=oho < H)

    H, HM, N, D = 16, 128, 128, 128
    torch.manual_seed(0)
    q_full = torch.zeros(HM, D, device='cuda', dtype=torch.bfloat16)
    q_full[:H] = torch.randn(H, D, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(N, D, device='cuda', dtype=torch.bfloat16)
    out = torch.zeros(H, N, device='cuda', dtype=torch.float32)

    try:
        toy[(1,)](q_full, k, out, H, HM, N, D, num_warps=4)
        torch.cuda.synchronize()
        ref = q_full[:H].float() @ k.float().t()
        err = (out - ref).abs().max().item()
        print(f"max abs err = {err:.3e}")

        # Now try slicing: does tmem.slice() work?
        print()
        print("Trying tmem slice-on-dim=1 too, with a new kernel that slices N")
        print("(omitted for brevity; main goal was to load+mask)")

    except Exception as e:
        import traceback
        traceback.print_exc()


@app.local_entrypoint()
def main():
    probe.remote()
