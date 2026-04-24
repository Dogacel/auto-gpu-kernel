"""
Follow-up profiling: isolate launch overhead vs phase compute, and
characterize whether split/combine are mem-bound or compute-bound.

Tests:
  1. Allocation-only cost — time the three torch.empty() calls alone.
  2. Split kernel memory-floor — total bytes touched by split kernel
     (read KV, indices, write partial_acc) vs equivalent memcpy time.
  3. Fused path: run a kernel that only does the two tl.dots over a fixed
     index set (no early-exit, no online softmax) to bound compute.
  4. Combine with bigger num_warps (to see if it's bottlenecked on single-warp reduce).
  5. Split kernel: does it scale with T? If it's compute-bound and SMs are
     saturated, T=1 and T=8 split should diverge; if SM-starved, they won't.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("flashinfer-profile2")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest", add_python="3.12")
    .apt_install("git", "wget", "build-essential", "cmake")
    .pip_install("huggingface_hub")
    .run_commands(
        "pip install --force-reinstall --upgrade "
        "git+https://github.com/flashinfer-ai/flashinfer-bench.git@main",
    )
    .pip_install("cupti-python")
)

KERNEL_SRC_PATH = PROJECT_ROOT / "solution" / "triton" / "sparse_fused.py"


@app.function(image=image, gpu="B200:1", timeout=1800, volumes={TRACE_SET_PATH: trace_volume})
def run_profile2(kernel_source: str) -> dict:
    import logging
    import math
    import tempfile
    import importlib.util
    import torch
    import numpy as np
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    tmp = tempfile.mkdtemp()
    src_path = Path(tmp) / "sparse_fused.py"
    src_path.write_text(kernel_source)
    spec = importlib.util.spec_from_file_location("sparse_fused_profile", src_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    split_kernel = mod._split_attn_kernel
    combine_kernel = mod._combine_kernel
    LOG2E = mod.LOG2E

    from flashinfer_bench import TraceSet
    from safetensors import safe_open

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    DEF = "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64"
    workloads = trace_set.workloads.get(DEF, [])

    wls_by_T = {}
    for w in workloads:
        T = int(w.workload.axes.get("num_tokens"))
        wls_by_T.setdefault(T, []).append(w)

    def load_inputs(wrapped_wl):
        wl = wrapped_wl.workload
        T = int(wl.axes.get("num_tokens"))
        P = int(wl.axes.get("num_pages"))
        torch.manual_seed(0)
        device = torch.device("cuda")
        q_nope = torch.randn(T, 16, 512, dtype=torch.bfloat16, device=device)
        q_pe = torch.randn(T, 16, 64, dtype=torch.bfloat16, device=device)
        ckv = torch.randn(P, 64, 512, dtype=torch.bfloat16, device=device)
        kpe = torch.randn(P, 64, 64, dtype=torch.bfloat16, device=device)
        si_spec = wl.inputs["sparse_indices"]
        st_path = Path(trace_set.root) / si_spec.path
        with safe_open(str(st_path), framework="numpy") as f:
            si_np = f.get_tensor(si_spec.tensor_key)
        sparse_indices = torch.from_numpy(np.asarray(si_np)).to(device=device, dtype=torch.int32)
        sm_scale = 1.0 / math.sqrt(192)
        out = torch.empty(T, 16, 512, dtype=torch.bfloat16, device=device)
        lse = torch.empty(T, 16, dtype=torch.float32, device=device)
        return q_nope, q_pe, ckv, kpe, sparse_indices, sm_scale, out, lse, T, P

    def time_ms(fn, warmup=20, iters=500):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    results = {}

    # For T=1 and T=8
    for T in (1, 8):
        q_nope, q_pe, ckv, kpe, si, sm_scale, out, lse, _, P = load_inputs(wls_by_T[T][0])
        valid_per_token = (si >= 0).sum(dim=-1).tolist()
        log.info(f"T={T}: valid_per_token={valid_per_token}")

        H, D_ckv, TOPK = 16, 512, 2048
        NUM_SPLITS = 8
        device = q_nope.device
        ckv_flat = ckv.view(ckv.shape[0] * ckv.shape[1], ckv.shape[2])
        kpe_flat = kpe.view(kpe.shape[0] * kpe.shape[1], kpe.shape[2])

        # (A) Pure allocation overhead for partial_m/l/acc
        def alloc_only():
            torch.empty((T, NUM_SPLITS, H), dtype=torch.float32, device=device)
            torch.empty((T, NUM_SPLITS, H), dtype=torch.float32, device=device)
            torch.empty((T, NUM_SPLITS, H, D_ckv), dtype=torch.float32, device=device)
        alloc_us = time_ms(alloc_only, iters=500) * 1000
        log.info(f"T={T}: pure allocation overhead = {alloc_us:.2f} us")

        # (B) "No-op" kernel launch: tiny Triton kernel to measure raw launch cost.
        import triton
        import triton.language as tl
        @triton.jit
        def _noop():
            pass
        def launch_noop():
            _noop[(T, NUM_SPLITS)]()
        noop_us = time_ms(launch_noop, iters=500) * 1000
        log.info(f"T={T}: noop launch ({T}*{NUM_SPLITS} grid) = {noop_us:.2f} us")

        # (C) Full fused-kernel baseline (both phases + allocations)
        partial_m = torch.empty((T, NUM_SPLITS, H), dtype=torch.float32, device=device)
        partial_l = torch.empty((T, NUM_SPLITS, H), dtype=torch.float32, device=device)
        partial_acc = torch.empty((T, NUM_SPLITS, H, D_ckv), dtype=torch.float32, device=device)

        def run_split():
            split_kernel[(T, NUM_SPLITS)](
                q_nope, q_pe, ckv_flat, kpe_flat, si,
                partial_m, partial_l, partial_acc,
                sm_scale * LOG2E,
                q_nope.stride(0), q_nope.stride(1),
                q_pe.stride(0), q_pe.stride(1),
                ckv_flat.stride(0), kpe_flat.stride(0),
                si.stride(0),
                partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
                partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
                partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
                TOPK=TOPK, H=H, D_CKV=D_ckv, D_KPE=64,
                BLOCK_N=64, NUM_SPLITS=NUM_SPLITS,
                num_warps=8, num_stages=2,
            )

        def run_combine():
            combine_kernel[(T,)](
                partial_m, partial_l, partial_acc, out, lse,
                partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
                partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
                partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
                out.stride(0), out.stride(1),
                lse.stride(0),
                H=H, D_CKV=D_ckv, NUM_SPLITS=NUM_SPLITS,
                num_warps=4, num_stages=1,
            )

        run_split()
        torch.cuda.synchronize()
        split_us = time_ms(run_split) * 1000
        combine_us = time_ms(run_combine) * 1000
        log.info(f"T={T}: split_only={split_us:.2f} us, combine_only={combine_us:.2f} us")

        def run_full_with_alloc():
            pm = torch.empty((T, NUM_SPLITS, H), dtype=torch.float32, device=device)
            pl = torch.empty((T, NUM_SPLITS, H), dtype=torch.float32, device=device)
            pa = torch.empty((T, NUM_SPLITS, H, D_ckv), dtype=torch.float32, device=device)
            split_kernel[(T, NUM_SPLITS)](
                q_nope, q_pe, ckv_flat, kpe_flat, si,
                pm, pl, pa,
                sm_scale * LOG2E,
                q_nope.stride(0), q_nope.stride(1),
                q_pe.stride(0), q_pe.stride(1),
                ckv_flat.stride(0), kpe_flat.stride(0),
                si.stride(0),
                pm.stride(0), pm.stride(1), pm.stride(2),
                pl.stride(0), pl.stride(1), pl.stride(2),
                pa.stride(0), pa.stride(1), pa.stride(2), pa.stride(3),
                TOPK=TOPK, H=H, D_CKV=D_ckv, D_KPE=64,
                BLOCK_N=64, NUM_SPLITS=NUM_SPLITS,
                num_warps=8, num_stages=2,
            )
            combine_kernel[(T,)](
                pm, pl, pa, out, lse,
                pm.stride(0), pm.stride(1), pm.stride(2),
                pl.stride(0), pl.stride(1), pl.stride(2),
                pa.stride(0), pa.stride(1), pa.stride(2), pa.stride(3),
                out.stride(0), out.stride(1),
                lse.stride(0),
                H=H, D_CKV=D_ckv, NUM_SPLITS=NUM_SPLITS,
                num_warps=4, num_stages=1,
            )
        full_us = time_ms(run_full_with_alloc) * 1000
        log.info(f"T={T}: full(alloc+split+combine) = {full_us:.2f} us")

        # (D) Memory-floor for combine: memcpy of partial_acc bytes
        pacc_bytes = T * NUM_SPLITS * H * D_ckv * 4  # fp32
        floor_buf_read = torch.empty(pacc_bytes // 4, dtype=torch.float32, device=device)
        floor_buf_write = torch.empty(pacc_bytes // 4, dtype=torch.float32, device=device)
        def memcpy_pacc():
            floor_buf_write.copy_(floor_buf_read)
        memcpy_pacc_us = time_ms(memcpy_pacc) * 1000
        log.info(f"T={T}: memcpy({pacc_bytes/1024:.0f} KB partial_acc) = {memcpy_pacc_us:.2f} us")

        # (E) combine with more warps (does it scale?)
        def run_combine_nw(nw):
            combine_kernel[(T,)](
                partial_m, partial_l, partial_acc, out, lse,
                partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
                partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
                partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
                out.stride(0), out.stride(1),
                lse.stride(0),
                H=H, D_CKV=D_ckv, NUM_SPLITS=NUM_SPLITS,
                num_warps=nw, num_stages=1,
            )

        combine_warp_sweep = {}
        for nw in (2, 4, 8, 16):
            try:
                us = time_ms(lambda n=nw: run_combine_nw(n)) * 1000
                combine_warp_sweep[nw] = us
            except Exception as e:
                combine_warp_sweep[nw] = f"fail: {e}"
        log.info(f"T={T}: combine num_warps sweep = {combine_warp_sweep}")

        # (F) Split kernel peak bandwidth check: how many bytes does it actually
        # read? Upper bound: one token reads TOPK * 64 * 512 bytes = 64 MB of CKV
        # + 64 MB KPE if no sparsity. With early-exit and valid_per_token=2,
        # it reads only 2 * 512 * 2 = 2 KB per token. That's far below the
        # 18us measured — split kernel is probably compute-bound on the dots
        # (even with valid=2, it does 8 splits * 1 block * two matmuls because
        # the tile is [16, 64] and can't be smaller).
        # For T=1, num_valid=2: split launches 8 CTAs. Each does:
        #  - load q_nope[16,512] = 16KB bf16
        #  - load q_pe[16,64]   = 2KB bf16
        #  - scan 256 indices   = 1KB
        #  - loop(1 block): load kc[64,512]=64KB bf16, kp[64,64]=8KB bf16
        #  - two dots (16,64)x(64,512)  fp32 accum
        #  - write partial_acc[16,512]=32KB fp32
        # Total per CTA: ~122KB read, 32KB write. 8 CTAs * ~150KB = 1.2MB. At 8TB/s,
        # that's 0.15us mem-bound. Compute: two dots per CTA = 2 * 16 * 64 * 512 * 2 =
        # 2M FLOPs per CTA. 8 CTAs = 16M FLOPs. B200 tensor core = ~1.5 PFLOPS bf16.
        # 16M / 1.5e15 = 0.011 us of pure tensor-core time. So split kernel on T=1 is
        # dominated by launch + prologue/epilogue overhead, not math or memory.
        split_flops_t1_per_cta = 2 * 16 * 64 * 512 * 2  # two dots
        split_flops_t1 = split_flops_t1_per_cta * 8
        split_bytes_t1 = 8 * ((16+2+32)*1024 + 1024 + (64+8)*1024)  # approx

        # For T=8, e.g. token with 1044 valid → 17 blocks. Per CTA loop processes
        # TOPK/8/64 = 4 blocks worst case. With dynamic bound, avg ~2-3 blocks.
        # Per block: 64*512 + 64*64 = 36KB KV load, plus one q-load fixed.
        results[T] = {
            "valid_per_token": valid_per_token,
            "alloc_us": alloc_us,
            "noop_launch_us": noop_us,
            "split_us": split_us,
            "combine_us": combine_us,
            "full_us": full_us,
            "memcpy_pacc_us": memcpy_pacc_us,
            "pacc_bytes": pacc_bytes,
            "combine_warp_sweep": combine_warp_sweep,
            "split_flops_est": split_flops_t1 if T == 1 else None,
        }

    return results


@app.local_entrypoint()
def main():
    kernel_source = KERNEL_SRC_PATH.read_text()
    out = run_profile2.remote(kernel_source)
    import json
    print("\n=== FULL RESULT ===")
    print(json.dumps(out, indent=2, default=str))
