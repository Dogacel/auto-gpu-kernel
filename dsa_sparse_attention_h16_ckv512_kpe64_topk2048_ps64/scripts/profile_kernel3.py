"""
Re-profile the current kernel (exp_9 structure).

Changes since Apr 16 profile:
  * exp_7: D-parallel combine kernel (grid `(T, D_CKV_SPLIT)`)
  * exp_8: D_CKV_SPLIT=8 (BLOCK_D=64)
  * exp_9: BLOCK_N=128 (was 64)

Questions this script answers:
  1. Split vs combine breakdown (percent + us) for T=1 (small) and T=8 (large)
  2. Per-phase compute vs memory ratio — bound shifted?
  3. HBM BW achieved (GB/s) vs B200 peak (~6-8 TB/s)
  4. SM occupancy / CTAs-in-flight per phase (from grid & kernel metadata)
  5. Roofline distance — speed-of-light for this problem
  6. Concrete bottleneck and top 3 levers

Launch:
    modal run scripts/profile_kernel3.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("flashinfer-profile3")

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
def run_profile3(kernel_source: str) -> dict:
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
    log.info(f"Loaded {len(workloads)} workloads")

    # B200 SM props
    props = torch.cuda.get_device_properties(0)
    NUM_SM = props.multi_processor_count  # B200 = 148
    log.info(f"Device: {props.name}, SMs={NUM_SM}, L2={props.L2_cache_size/1024:.0f} KB")
    # Peak FLOPS: B200 bf16 tensor core ~2.25 PFLOPS dense. HBM3e ~8 TB/s.
    PEAK_BF16_TFLOPS = 2250.0    # B200 dense bf16 TC theoretical
    PEAK_HBM_GBS = 8000.0        # B200 HBM3e peak

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

    def time_with_events(fn, warmup=20, iters=500):
        """Return (p50_us, p90_us) of distribution."""
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        xs = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            xs.append(s.elapsed_time(e) * 1000)  # us
        xs.sort()
        return xs[int(0.5 * len(xs))], xs[int(0.9 * len(xs))]

    # Current config (exp_9)
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
        "config": {
            "BLOCK_N": BLOCK_N,
            "NUM_SPLITS": NUM_SPLITS,
            "D_CKV_SPLIT": D_CKV_SPLIT,
            "BLOCK_D": BLOCK_D,
            "H": H,
            "D_CKV": D_CKV,
            "D_KPE": D_KPE,
            "TOPK": TOPK,
        },
        "per_T": {},
    }

    for T in (1, 8):
        q_nope, q_pe, ckv, kpe, si, sm_scale, out, lse, _, P = load_inputs(wls_by_T[T][0])
        valid_per_token = (si >= 0).sum(dim=-1).tolist()
        total_valid = int((si >= 0).sum().item())
        log.info(f"=== T={T}: valid_per_token={valid_per_token}, total={total_valid} ===")

        device = q_nope.device
        ckv_flat = ckv.view(ckv.shape[0] * ckv.shape[1], ckv.shape[2])
        kpe_flat = kpe.view(kpe.shape[0] * kpe.shape[1], kpe.shape[2])

        # Allocate partials (shared across split/combine timings)
        partial_m = torch.empty((T, NUM_SPLITS, H), dtype=torch.float32, device=device)
        partial_l = torch.empty((T, NUM_SPLITS, H), dtype=torch.float32, device=device)
        partial_acc = torch.empty((T, NUM_SPLITS, H, D_CKV), dtype=torch.float32, device=device)

        # ------ Kernel launch helpers using current config ------
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

        # Prime partial_acc with real values
        run_split()
        torch.cuda.synchronize()

        # (1) split & combine standalone (mean) + percentiles
        split_p50, split_p90 = time_with_events(run_split)
        combine_p50, combine_p90 = time_with_events(run_combine)
        log.info(f"T={T}: split_p50={split_p50:.2f} us (p90 {split_p90:.2f})")
        log.info(f"T={T}: combine_p50={combine_p50:.2f} us (p90 {combine_p90:.2f})")

        # (2) Full fused path (alloc + launch + launch) with events
        def run_full():
            pm = torch.empty((T, NUM_SPLITS, H), dtype=torch.float32, device=device)
            pl = torch.empty((T, NUM_SPLITS, H), dtype=torch.float32, device=device)
            pa = torch.empty((T, NUM_SPLITS, H, D_CKV), dtype=torch.float32, device=device)
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
                TOPK=TOPK, H=H, D_CKV=D_CKV, D_KPE=D_KPE,
                BLOCK_N=BLOCK_N, NUM_SPLITS=NUM_SPLITS,
                num_warps=8, num_stages=2,
            )
            combine_kernel[(T, D_CKV_SPLIT)](
                pm, pl, pa, out, lse,
                pm.stride(0), pm.stride(1), pm.stride(2),
                pl.stride(0), pl.stride(1), pl.stride(2),
                pa.stride(0), pa.stride(1), pa.stride(2), pa.stride(3),
                out.stride(0), out.stride(1),
                lse.stride(0),
                H=H, D_CKV=D_CKV, NUM_SPLITS=NUM_SPLITS, BLOCK_D=BLOCK_D,
                num_warps=4, num_stages=1,
            )
        full_p50, full_p90 = time_with_events(run_full)
        log.info(f"T={T}: full (event, alloc+split+combine) p50={full_p50:.2f} us / p90={full_p90:.2f}")

        # (3) Pure alloc cost
        def alloc_only():
            torch.empty((T, NUM_SPLITS, H), dtype=torch.float32, device=device)
            torch.empty((T, NUM_SPLITS, H), dtype=torch.float32, device=device)
            torch.empty((T, NUM_SPLITS, H, D_CKV), dtype=torch.float32, device=device)
        alloc_us = time_ms(alloc_only) * 1000

        # (4) Noop launch (raw Triton launch cost)
        @triton.jit
        def _noop_split():
            pass
        @triton.jit
        def _noop_combine():
            pass
        def launch_noops():
            _noop_split[(T, NUM_SPLITS)]()
            _noop_combine[(T, D_CKV_SPLIT)]()
        noop_us = time_ms(launch_noops) * 1000
        log.info(f"T={T}: alloc={alloc_us:.2f} us, noop(2 launches)={noop_us:.2f} us")

        # (5) Memory-floor anchor — memcpy of partial_acc volume (combine input)
        pacc_bytes = T * NUM_SPLITS * H * D_CKV * 4
        buf_r = torch.empty(pacc_bytes // 4, dtype=torch.float32, device=device)
        buf_w = torch.empty(pacc_bytes // 4, dtype=torch.float32, device=device)
        def memcpy_pacc():
            buf_w.copy_(buf_r)
        memcpy_pacc_us = time_ms(memcpy_pacc) * 1000
        log.info(f"T={T}: memcpy({pacc_bytes/1024:.0f} KB)={memcpy_pacc_us:.2f} us")

        # Split-kernel memory anchor: bytes read of kv + indices + qs, bytes written
        # for a typical split, per T=1 vs T=8. Using valid counts.
        # For T=8: split kernel per-CTA processes ceil(valid_in_split / BLOCK_N) blocks,
        #   each block loads 128 * (512+64) bytes bf16 = 73728 bytes KV per block
        # Total across all (T, NUM_SPLITS) CTAs
        est_split_bytes = 0
        for t_idx, v in enumerate(valid_per_token):
            per_split = (v + NUM_SPLITS - 1) // NUM_SPLITS  # upper bound valid per split
            # each split rounds up to BLOCK_N
            for s in range(NUM_SPLITS):
                # how many valid in this specific split?
                start = s * (TOPK // NUM_SPLITS)
                end = start + TOPK // NUM_SPLITS
                vs = max(0, min(v, end) - start) if v > start else 0
                blocks = (vs + BLOCK_N - 1) // BLOCK_N
                est_split_bytes += blocks * BLOCK_N * (D_CKV + D_KPE) * 2  # bf16
        # Add Q load: T * (NUM_SPLITS CTAs each reads Q_nope + Q_pe)
        q_bytes_per_cta = H * (D_CKV + D_KPE) * 2  # bf16
        est_split_bytes += T * NUM_SPLITS * q_bytes_per_cta
        # Indices read (approx)
        est_split_bytes += T * NUM_SPLITS * (TOPK // NUM_SPLITS) * 4  # i32
        # Partial_acc writes
        est_split_bytes += T * NUM_SPLITS * H * D_CKV * 4  # fp32 out

        split_bw_achieved_gbs = est_split_bytes / 1e9 / (split_p50 / 1e6)
        log.info(f"T={T}: split est_bytes={est_split_bytes/1024/1024:.2f} MB, BW={split_bw_achieved_gbs:.1f} GB/s")

        # Combine kernel bytes: reads partial_acc + partial_m/l, writes output + lse
        combine_read_bytes = (T * NUM_SPLITS * H * D_CKV * 4) + (T * NUM_SPLITS * 2 * H * 4)
        combine_write_bytes = (T * H * D_CKV * 2) + (T * H * 4)
        combine_bytes = combine_read_bytes + combine_write_bytes
        combine_bw_achieved_gbs = combine_bytes / 1e9 / (combine_p50 / 1e6)
        log.info(f"T={T}: combine bytes={combine_bytes/1024:.0f} KB, BW={combine_bw_achieved_gbs:.1f} GB/s")

        # (6) Compute throughput for split kernel (tensor-core)
        # Per iter: two dots
        #   logits = q_nope [H, D_CKV] @ kc.T [D_CKV, BLOCK_N]   → 2 * H * D_CKV * BLOCK_N FLOPs
        #   logits += q_pe [H, D_KPE] @ kp.T [D_KPE, BLOCK_N]    → 2 * H * D_KPE * BLOCK_N FLOPs
        #   acc += p [H, BLOCK_N] @ kc [BLOCK_N, D_CKV]          → 2 * H * BLOCK_N * D_CKV FLOPs
        flops_per_block = 2 * H * D_CKV * BLOCK_N + 2 * H * D_KPE * BLOCK_N + 2 * H * BLOCK_N * D_CKV
        total_split_flops = 0
        for v in valid_per_token:
            # iterate over all splits
            for s in range(NUM_SPLITS):
                start = s * (TOPK // NUM_SPLITS)
                end = start + TOPK // NUM_SPLITS
                vs = max(0, min(v, end) - start) if v > start else 0
                blocks = (vs + BLOCK_N - 1) // BLOCK_N
                total_split_flops += blocks * flops_per_block

        split_tflops_achieved = total_split_flops / 1e12 / (split_p50 / 1e6)
        # Roofline: split kernel arithmetic intensity
        split_ai = total_split_flops / est_split_bytes if est_split_bytes else 0.0
        # B200 balance point: 2250 TFLOPS / 8 TB/s = 281 FLOPs/byte
        bw_bound_time_us = est_split_bytes / (PEAK_HBM_GBS * 1e9) * 1e6
        compute_bound_time_us = total_split_flops / (PEAK_BF16_TFLOPS * 1e12) * 1e6
        sol_us = max(bw_bound_time_us, compute_bound_time_us)
        log.info(f"T={T}: split FLOPs={total_split_flops/1e9:.2f} GFLOPs, achieved={split_tflops_achieved:.2f} TFLOPS")
        log.info(f"T={T}: split AI={split_ai:.1f} FLOPs/byte, BW-bound SoL={bw_bound_time_us:.2f}us, Compute-bound SoL={compute_bound_time_us:.2f}us, chosen SoL={sol_us:.2f}us")

        # (7) Combine compute
        # Per iter: merge alpha, beta, acc = acc*alpha + acc_si*beta — 2 * H * BLOCK_D FMAs
        # Plus 2 exp2 (H each), some maximum/broadcast ops. Total compute is small.
        combine_flops = T * D_CKV_SPLIT * NUM_SPLITS * (2 * 2 * H * BLOCK_D + 4 * H)  # FMAs across mul/add + ~4 ops/head/iter overhead
        # Normalize per-CTA
        combine_tflops_achieved = combine_flops / 1e12 / (combine_p50 / 1e6)
        combine_bw_bound_us = combine_bytes / (PEAK_HBM_GBS * 1e9) * 1e6
        combine_compute_bound_us = combine_flops / (PEAK_BF16_TFLOPS * 1e12) * 1e6
        combine_ai = combine_flops / combine_bytes if combine_bytes else 0.0
        combine_sol_us = max(combine_bw_bound_us, combine_compute_bound_us)
        log.info(f"T={T}: combine AI={combine_ai:.1f}, BW-bound SoL={combine_bw_bound_us:.2f}us, Compute-bound SoL={combine_compute_bound_us:.2f}us")

        # (8) CTAs in flight (occupancy proxy)
        split_grid = T * NUM_SPLITS
        combine_grid = T * D_CKV_SPLIT
        # B200 has 148 SMs. CTAs in flight ≤ min(grid, NUM_SM * cta_per_sm).
        # We don't have exact regs from static metadata; assume cta_per_sm >= 1 for split, >= 2 for combine.
        # At minimum, occupancy = grid / NUM_SM
        split_occ = split_grid / NUM_SM
        combine_occ = combine_grid / NUM_SM

        # (9) Inspect sparse index validity distribution for small vs large workloads
        # Derive time-in-hot-loop estimate: blocks * (compute + mem per block)
        # For T=1 small with valid=2: hot loop is 1 block × 8 CTAs — mostly prologue
        # For T=8 large: varied. Let's extract worst-valid token distribution across the 8 splits.
        per_split_valids = [[0] * NUM_SPLITS for _ in range(T)]
        for t_idx in range(T):
            v = valid_per_token[t_idx]
            for s in range(NUM_SPLITS):
                start = s * (TOPK // NUM_SPLITS)
                end = start + TOPK // NUM_SPLITS
                vs = max(0, min(v, end) - start) if v > start else 0
                per_split_valids[t_idx][s] = vs

        max_block_count = 0
        for row in per_split_valids:
            for vs in row:
                max_block_count = max(max_block_count, (vs + BLOCK_N - 1) // BLOCK_N)
        log.info(f"T={T}: max blocks per CTA (worst straggler) = {max_block_count}")

        results["per_T"][T] = {
            "valid_per_token": valid_per_token,
            "total_valid": total_valid,
            "max_blocks_per_cta": max_block_count,
            "split_p50_us": split_p50,
            "split_p90_us": split_p90,
            "combine_p50_us": combine_p50,
            "combine_p90_us": combine_p90,
            "full_p50_us": full_p50,
            "full_p90_us": full_p90,
            "alloc_us": alloc_us,
            "noop_us": noop_us,
            "memcpy_pacc_us": memcpy_pacc_us,
            "pacc_bytes": pacc_bytes,
            "est_split_bytes": est_split_bytes,
            "split_bw_achieved_gbs": split_bw_achieved_gbs,
            "total_split_flops": total_split_flops,
            "split_tflops_achieved": split_tflops_achieved,
            "split_ai_flops_per_byte": split_ai,
            "split_bw_bound_sol_us": bw_bound_time_us,
            "split_compute_bound_sol_us": compute_bound_time_us,
            "split_sol_us": sol_us,
            "combine_bytes": combine_bytes,
            "combine_bw_achieved_gbs": combine_bw_achieved_gbs,
            "combine_flops": combine_flops,
            "combine_tflops_achieved": combine_tflops_achieved,
            "combine_bw_bound_sol_us": combine_bw_bound_us,
            "combine_compute_bound_sol_us": combine_compute_bound_us,
            "combine_sol_us": combine_sol_us,
            "split_grid": split_grid,
            "combine_grid": combine_grid,
            "split_occupancy_pct": split_occ * 100,
            "combine_occupancy_pct": combine_occ * 100,
        }

    # (10) Broad sweep across all workloads — distribution of full latency
    all_full_us = []
    all_by_T = {}
    for wrapped_wl in workloads:
        q_nope, q_pe, ckv, kpe, si, sm_scale, out, lse, T, P = load_inputs(wrapped_wl)
        device = q_nope.device
        ckv_flat = ckv.view(ckv.shape[0] * ckv.shape[1], ckv.shape[2])
        kpe_flat = kpe.view(kpe.shape[0] * kpe.shape[1], kpe.shape[2])
        partial_m = torch.empty((T, NUM_SPLITS, H), dtype=torch.float32, device=device)
        partial_l = torch.empty((T, NUM_SPLITS, H), dtype=torch.float32, device=device)
        partial_acc = torch.empty((T, NUM_SPLITS, H, D_CKV), dtype=torch.float32, device=device)
        total_v = int((si >= 0).sum().item())

        def run_full_ts():
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
        us = time_ms(run_full_ts, iters=200) * 1000
        all_full_us.append((wrapped_wl.workload.uuid[:8], T, total_v, us))
        all_by_T.setdefault(T, []).append(us)

    # Sort worst-5 by absolute us & by us/valid
    worst_abs = sorted(all_full_us, key=lambda x: -x[3])[:5]
    # us per valid token (rough proxy for per-work)
    ratio = [(uuid, T, v, u, u / max(v, 1)) for (uuid, T, v, u) in all_full_us]
    worst_ratio = sorted(ratio, key=lambda x: -x[4])[:5]
    results["workload_sweep"] = {
        "all": all_full_us,
        "worst5_abs_us": worst_abs,
        "worst5_us_per_valid": worst_ratio,
    }
    log.info(f"Worst-5 by absolute us: {worst_abs}")
    log.info(f"Worst-5 by us/valid: {worst_ratio}")

    return results


@app.local_entrypoint()
def main():
    kernel_source = KERNEL_SRC_PATH.read_text()
    print(f"Profiling {KERNEL_SRC_PATH} (current exp_9 structure) on Modal B200...")
    out = run_profile3.remote(kernel_source)
    import json
    print("\n=== FULL RESULT ===")
    print(json.dumps(out, indent=2, default=str))
