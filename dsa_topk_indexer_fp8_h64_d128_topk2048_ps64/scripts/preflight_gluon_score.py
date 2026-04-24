"""Standalone compile/correctness test for the Gluon score_kernel port. Faster than
round-tripping through flashinfer-bench; runs score_kernel against a tiny synthetic
input and compares against the pytorch reference semantics."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("preflight-gluon-score")

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest", add_python="3.12")
    .apt_install("git", "wget", "build-essential", "cmake")
    .add_local_dir(
        str(PROJECT_ROOT / "solution" / "triton"),
        "/root/user_kernels",
    )
)


@app.function(image=image, gpu="B200", timeout=600)
def test():
    import sys
    sys.path.insert(0, "/root/user_kernels")
    import torch
    import triton

    print(f"Triton: {triton.__version__}")

    # Import the kernel module
    from indexer_fused import score_kernel

    # Build synthetic inputs matching the benchmark shape
    torch.manual_seed(0)
    B = 4
    mp = 3  # max_num_pages; needs > 32 for score_kernel path; but we invoke score_kernel directly
    H = 64
    D = 128
    page_size = 64
    num_pages_total = 8  # 8 pages total in cache
    max_scored = mp * page_size

    # Q: [B, H, D] fp8
    q_f = torch.randn(B, H, D, device="cuda", dtype=torch.float32) * 0.5
    q_fp8 = q_f.to(torch.float8_e4m3fn)

    # K index cache: [P, page_size, 1, 132] int8 (SOA: fp8 then scales)
    page_bytes = page_size * (D + 4)  # 132 fp8 units per "row"; but actually 8192 fp8 + 256 scale bytes
    # Easier: just build fp8_view and scale_view directly
    fp8_data = (torch.randn(num_pages_total, page_size, D, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    scales = torch.rand(num_pages_total, page_size, device="cuda", dtype=torch.float32) * 0.5 + 0.1

    # Pack into the int8 byte layout
    total_bytes_per_page = page_size * D + page_size * 4
    packed = torch.zeros(num_pages_total, total_bytes_per_page, device="cuda", dtype=torch.uint8)
    packed[:, :page_size * D] = fp8_data.view(num_pages_total, page_size * D).view(torch.uint8)
    packed[:, page_size * D:] = scales.reshape(num_pages_total, page_size).view(torch.uint8).view(num_pages_total, -1)

    # As the actual kernel uses: fp8_view[p, t, d] and scale_view[p, t]
    fp8_view = torch.as_strided(
        packed.view(torch.float8_e4m3fn),
        size=(num_pages_total, page_size, D),
        stride=(total_bytes_per_page, D, 1),
    )
    scale_view = torch.as_strided(
        packed.view(torch.float32),
        size=(num_pages_total, page_size),
        stride=(total_bytes_per_page // 4, 1),
        storage_offset=page_size * D // 4,
    )

    # weights: [B, H] fp32
    weights = (torch.randn(B, H, device="cuda") * 0.5 + 1.0)

    # seq_lens: vary so we hit both paths
    seq_lens = torch.tensor([mp * page_size, mp * page_size - 10, 1 * page_size + 5, 2 * page_size],
                            device="cuda", dtype=torch.int32)

    # block_table: [B, mp] int32 -- random page ids
    block_table = torch.randint(0, num_pages_total, (B, mp), device="cuda", dtype=torch.int32)

    # scores output: [B, max_scored] fp32
    scores = torch.empty(B, max_scored, device="cuda", dtype=torch.float32)

    # Launch
    grid = (B, mp)
    try:
        score_kernel[grid](
            q_fp8, fp8_view, scale_view, weights, seq_lens, block_table, scores,
            q_fp8.stride(0), q_fp8.stride(1), q_fp8.stride(2),
            fp8_view.stride(0), fp8_view.stride(1), fp8_view.stride(2),
            scale_view.stride(0), scale_view.stride(1),
            weights.stride(0), weights.stride(1),
            block_table.stride(0), block_table.stride(1),
            scores.stride(0), scores.stride(1),
            BLOCK_H=H, BLOCK_D=D, BLOCK_T=page_size,
        )
        torch.cuda.synchronize()
        print("Gluon score_kernel compiled & ran OK!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nFAILED: {type(e).__name__}: {e}")
        return

    # Correctness: compute the reference the same way the Python ref does.
    q_f32 = q_fp8.to(torch.float32)
    scores_ref = torch.full((B, max_scored), -1e30, device="cuda", dtype=torch.float32)
    for b in range(B):
        sl = int(seq_lens[b].item())
        if sl == 0:
            continue
        pages_needed = (sl + page_size - 1) // page_size
        page_ids = block_table[b, :pages_needed].to(torch.long)
        K_pages = fp8_view[page_ids].to(torch.float32)  # [pages_needed, page_size, D]
        S_pages = scale_view[page_ids]  # [pages_needed, page_size]
        K = K_pages.reshape(-1, D)[:sl]
        S = S_pages.reshape(-1)[:sl]
        q_b = q_f32[b]
        dots = q_b @ K.T  # [H, sl]
        dots = torch.relu(dots)
        w = weights[b]
        weighted = dots * w[:, None]  # [H, sl]
        final = weighted.sum(dim=0) * S  # [sl]
        scores_ref[b, :sl] = final
        # Beyond sl-within-pages, the kernel also writes -1e30; leave as is.

    # Compare
    diff = (scores - scores_ref).abs()
    # Focus on the valid region where the kernel should match
    for b in range(B):
        sl = int(seq_lens[b].item())
        kernel_slice = scores[b, :sl]
        ref_slice = scores_ref[b, :sl]
        if sl > 0:
            err = (kernel_slice - ref_slice).abs().max().item()
            mean_err = (kernel_slice - ref_slice).abs().mean().item()
            print(f"  B={b}, sl={sl}, max_abs_err={err:.4e}, mean_abs_err={mean_err:.4e}")


@app.local_entrypoint()
def run():
    test.remote()
