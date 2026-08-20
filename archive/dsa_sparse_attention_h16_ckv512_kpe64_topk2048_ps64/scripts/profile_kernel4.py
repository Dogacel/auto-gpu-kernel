"""
Cupti-calibrated phase timing for current kernel (exp_9).

profile_kernel3 used per-call Event timing, which adds ~4us per call overhead on
tiny kernels. This harness uses cuda graph capture + Event timing of a BLOCK of
iterations (amortizing event overhead across many calls) so per-call cost is
accurate.

Actually — NOTE: project rule says no CUDA graphs. So we bracket a LOOP of N
iterations with a single Event pair, then divide by N. Event-insertion overhead
is paid ONCE, not per iteration.

Launch:
    modal run scripts/profile_kernel4.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("flashinfer-profile4")

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
def run_profile4(kernel_source: str) -> dict:
    import logging
    import math
    import tempfile
    import importlib.util
    import torch
    import numpy as np
    import triton
    import triton.language as tl
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

    props = torch.cuda.get_device_properties(0)
    NUM_SM = props.multi_processor_count
    PEAK_BF16_TFLOPS = 2250.0
    PEAK_HBM_GBS = 8000.0

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

    def time_per_call_us(fn, warmup=50, inner=200, outer=50):
        """Event pair brackets `inner` calls; repeat `outer` times; return p50 & p90 per-call us."""
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        xs = []
        for _ in range(outer):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(inner):
                fn()
            e.record()
            torch.cuda.synchronize()
            xs.append(s.elapsed_time(e) * 1000 / inner)
        xs.sort()
        return xs[int(0.5 * len(xs))], xs[int(0.9 * len(xs))]

    BLOCK_N = 128
    NUM_SPLITS = 8
    D_CKV_SPLIT = 8
    H = 16
    D_CKV = 512
    D_KPE = 64
    TOPK = 2048
    BLOCK_D = D_CKV // D_CKV_SPLIT

    results = {
        "device": props.name,
        "num_sm": NUM_SM,
        "peak_bf16_tflops": PEAK_BF16_TFLOPS,
        "peak_hbm_gbs": PEAK_HBM_GBS,
        "per_T": {},
    }

    # Use an empty Triton kernel for launch-overhead baseline
    @triton.jit
    def _noop():
        pass

    for T in (1, 8):
        q_nope, q_pe, ckv, kpe, si, sm_scale, out, lse, _, P = load_inputs(wls_by_T[T][0])
        valid_per_token = (si >= 0).sum(dim=-1).tolist()
        total_valid = int((si >= 0).sum().item())
        log.info(f"=== T={T}: valid={valid_per_token}, total={total_valid} ===")

        device = q_nope.device
        ckv_flat = ckv.view(ckv.shape[0] * ckv.shape[1], ckv.shape[2])
        kpe_flat = kpe.view(kpe.shape[0] * kpe.shape[1], kpe.shape[2])

        partial_m = torch.empty((T, NUM_SPLITS, H), dtype=torch.float32, device=device)
        partial_l = torch.empty((T, NUM_SPLITS, H), dtype=torch.float32, device=device)
        partial_acc = torch.empty((T, NUM_SPLITS, H, D_CKV), dtype=torch.float32, device=device)

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
                TOPK=TOPK, H=H, D_CKV=D_CKV, D_KPE=D_KPE,
                BLOCK_N=BLOCK_N, NUM_SPLITS=NUM_SPLITS,
                num_warps=8, num_stages=2,
            )

        def run_combine():
            combine_kernel[(T, D_CKV_SPLIT)](
                partial_m, partial_l, partial_acc, out, lse,
                partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
                partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
                partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
                out.stride(0), out.stride(1),
                lse.stride(0),
                H=H, D_CKV=D_CKV, NUM_SPLITS=NUM_SPLITS, BLOCK_D=BLOCK_D,
                num_warps=4, num_stages=1,
            )

        run_split()
        torch.cuda.synchronize()

        # Amortized per-call timing
        split_p50, split_p90 = time_per_call_us(run_split)
        combine_p50, combine_p90 = time_per_call_us(run_combine)

        def run_full():
            run_split()
            run_combine()
        full_p50, full_p90 = time_per_call_us(run_full)

        def run_noop():
            _noop[(T, NUM_SPLITS)]()
        noop_p50, noop_p90 = time_per_call_us(run_noop)

        def run_two_noops():
            _noop[(T, NUM_SPLITS)]()
            _noop[(T, D_CKV_SPLIT)]()
        two_noop_p50, _ = time_per_call_us(run_two_noops)

        # Memcpy-floor anchor for combine
        pacc_bytes = T * NUM_SPLITS * H * D_CKV * 4
        buf_r = torch.empty(pacc_bytes // 4, dtype=torch.float32, device=device)
        buf_w = torch.empty(pacc_bytes // 4, dtype=torch.float32, device=device)
        def memcpy_pacc():
            buf_w.copy_(buf_r)
        memcpy_p50, _ = time_per_call_us(memcpy_pacc)

        log.info(f"T={T}: split p50={split_p50:.2f}us p90={split_p90:.2f}us")
        log.info(f"T={T}: combine p50={combine_p50:.2f}us p90={combine_p90:.2f}us")
        log.info(f"T={T}: full p50={full_p50:.2f}us p90={full_p90:.2f}us")
        log.info(f"T={T}: single noop launch={noop_p50:.2f}us, two noops={two_noop_p50:.2f}us")
        log.info(f"T={T}: memcpy(partial_acc)={memcpy_p50:.2f}us")

        # Achieved bandwidth calculations
        # Split kernel bytes (accurate accounting)
        est_split_bytes = 0
        for v in valid_per_token:
            for s in range(NUM_SPLITS):
                start = s * (TOPK // NUM_SPLITS)
                end = start + TOPK // NUM_SPLITS
                vs = max(0, min(v, end) - start) if v > start else 0
                blocks = (vs + BLOCK_N - 1) // BLOCK_N
                # Per block: load BLOCK_N positions of ckv (D_CKV=512 bf16) + kpe (D_KPE=64 bf16)
                est_split_bytes += blocks * BLOCK_N * (D_CKV + D_KPE) * 2
        # Q load per CTA: H * (D_CKV + D_KPE) * 2 bytes bf16
        q_bytes_per_cta = H * (D_CKV + D_KPE) * 2
        est_split_bytes += T * NUM_SPLITS * q_bytes_per_cta
        # Indices scan
        est_split_bytes += T * NUM_SPLITS * (TOPK // NUM_SPLITS) * 4
        # Partial_acc writes (fp32)
        est_split_bytes += T * NUM_SPLITS * H * D_CKV * 4

        # Split compute (tensor-core FLOPs)
        flops_per_block = 2 * H * D_CKV * BLOCK_N + 2 * H * D_KPE * BLOCK_N + 2 * H * BLOCK_N * D_CKV
        total_split_flops = 0
        for v in valid_per_token:
            for s in range(NUM_SPLITS):
                start = s * (TOPK // NUM_SPLITS)
                end = start + TOPK // NUM_SPLITS
                vs = max(0, min(v, end) - start) if v > start else 0
                blocks = (vs + BLOCK_N - 1) // BLOCK_N
                total_split_flops += blocks * flops_per_block

        split_bw_gbs = est_split_bytes / 1e9 / (split_p50 / 1e6)
        split_tflops = total_split_flops / 1e12 / (split_p50 / 1e6)
        split_ai = total_split_flops / est_split_bytes if est_split_bytes else 0.0
        split_bw_bound_us = est_split_bytes / (PEAK_HBM_GBS * 1e9) * 1e6
        split_compute_bound_us = total_split_flops / (PEAK_BF16_TFLOPS * 1e12) * 1e6
        split_sol_us = max(split_bw_bound_us, split_compute_bound_us)

        # Combine bytes: read partial_acc + partial_m/l + write out + lse
        combine_bytes = (T * NUM_SPLITS * H * D_CKV * 4) + (T * NUM_SPLITS * 2 * H * 4) + (T * H * D_CKV * 2) + (T * H * 4)
        # Combine compute (FMAs): NUM_SPLITS iters × [2*H*BLOCK_D for acc*alpha + acc_si*beta] × grid size
        combine_flops = T * D_CKV_SPLIT * NUM_SPLITS * 2 * 2 * H * BLOCK_D  # Rough
        combine_bw_gbs = combine_bytes / 1e9 / (combine_p50 / 1e6)
        combine_tflops = combine_flops / 1e12 / (combine_p50 / 1e6)
        combine_bw_bound_us = combine_bytes / (PEAK_HBM_GBS * 1e9) * 1e6
        combine_compute_bound_us = combine_flops / (PEAK_BF16_TFLOPS * 1e12) * 1e6
        combine_sol_us = max(combine_bw_bound_us, combine_compute_bound_us)

        split_grid = T * NUM_SPLITS
        combine_grid = T * D_CKV_SPLIT

        results["per_T"][T] = {
            "valid_per_token": valid_per_token,
            "total_valid": total_valid,
            "split_p50_us": split_p50,
            "split_p90_us": split_p90,
            "combine_p50_us": combine_p50,
            "combine_p90_us": combine_p90,
            "full_p50_us": full_p50,
            "full_p90_us": full_p90,
            "noop_single_us": noop_p50,
            "two_noop_us": two_noop_p50,
            "memcpy_pacc_us": memcpy_p50,
            "pacc_bytes": pacc_bytes,
            "est_split_bytes": est_split_bytes,
            "total_split_flops": total_split_flops,
            "split_bw_gbs": split_bw_gbs,
            "split_tflops": split_tflops,
            "split_ai_flops_per_byte": split_ai,
            "split_bw_bound_us": split_bw_bound_us,
            "split_compute_bound_us": split_compute_bound_us,
            "split_sol_us": split_sol_us,
            "combine_bytes": combine_bytes,
            "combine_flops": combine_flops,
            "combine_bw_gbs": combine_bw_gbs,
            "combine_tflops": combine_tflops,
            "combine_bw_bound_us": combine_bw_bound_us,
            "combine_compute_bound_us": combine_compute_bound_us,
            "combine_sol_us": combine_sol_us,
            "split_grid": split_grid,
            "combine_grid": combine_grid,
            "split_occupancy_pct": 100.0 * split_grid / NUM_SM,
            "combine_occupancy_pct": 100.0 * combine_grid / NUM_SM,
        }

    return results


@app.local_entrypoint()
def main():
    kernel_source = KERNEL_SRC_PATH.read_text()
    print(f"Profiling (cupti-equivalent timing) on Modal B200...")
    out = run_profile4.remote(kernel_source)
    import json
    print("\n=== FULL RESULT ===")
    print(json.dumps(out, indent=2, default=str))
