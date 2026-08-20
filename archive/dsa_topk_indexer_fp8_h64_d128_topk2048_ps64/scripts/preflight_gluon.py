"""Attempt to use the translator on score_kernel, and benchmark the auto-translated Gluon version."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("preflight-gluon")

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest", add_python="3.12")
    .apt_install("git", "wget", "build-essential", "cmake")
)


@app.function(image=image, gpu="B200", timeout=600)
def translate_score_kernel():
    """Run the Triton -> Gluon translator on our score_kernel and print the emitted Gluon source."""
    import textwrap
    import triton
    import triton.language as tl
    from triton.tools.triton_to_gluon_translater.translator import convert_triton_to_gluon

    @triton.jit
    def score_kernel(
        q_ptr, k_fp8_ptr, k_scale_ptr, w_ptr,
        seq_lens_ptr, block_table_ptr, scores_ptr,
        stride_qb, stride_qh, stride_qd,
        stride_kp, stride_kt, stride_kd,
        stride_ksp, stride_kst,
        stride_wb, stride_wh,
        stride_btb, stride_btp,
        stride_sb, stride_st,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_p = tl.program_id(1)

        seq_len = tl.load(seq_lens_ptr + pid_b)
        token_start = pid_p * BLOCK_T

        if token_start >= seq_len:
            t_offs_sk = tl.arange(0, BLOCK_T)
            score_off_sk = pid_b * stride_sb + (token_start + t_offs_sk) * stride_st
            tl.store(scores_ptr + score_off_sk, tl.full([BLOCK_T], -1e30, tl.float32))
            return

        page_id_raw = tl.load(block_table_ptr + pid_b * stride_btb + pid_p * stride_btp)
        page_id = page_id_raw.to(tl.int64)

        h_offs = tl.arange(0, BLOCK_H)
        d_offs = tl.arange(0, BLOCK_D)
        t_offs = tl.arange(0, BLOCK_T)

        q_off = pid_b * stride_qb + h_offs[:, None] * stride_qh + d_offs[None, :] * stride_qd
        q_fp8 = tl.load(q_ptr + q_off)

        k_off = page_id * stride_kp + t_offs[:, None] * stride_kt + d_offs[None, :] * stride_kd
        k_fp8 = tl.load(k_fp8_ptr + k_off)

        s_off = page_id * stride_ksp + t_offs * stride_kst
        scale = tl.load(k_scale_ptr + s_off)
        w_off = pid_b * stride_wb + h_offs * stride_wh
        w = tl.load(w_ptr + w_off)

        scores = tl.dot(q_fp8, tl.trans(k_fp8), out_dtype=tl.float32)
        scores = tl.maximum(scores, 0.0)
        scores = scores * w[:, None]
        final = tl.sum(scores, axis=0) * scale

        abs_t = token_start + t_offs
        in_bounds = abs_t < seq_len
        final = tl.where(in_bounds, final, -1e30)

        score_off = pid_b * stride_sb + abs_t * stride_st
        tl.store(scores_ptr + score_off, final)

    gluon_src = convert_triton_to_gluon([score_kernel])
    print("=========== AUTO-TRANSLATED GLUON SOURCE ===========")
    print(gluon_src)
    print("=========== END ===========")


@app.local_entrypoint()
def run():
    translate_score_kernel.remote()
