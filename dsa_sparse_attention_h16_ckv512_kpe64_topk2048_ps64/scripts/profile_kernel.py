"""
Profiling harness for the two-phase split-K DSA sparse attention kernel.

Measures, per workload (T=1 and T=8 specifically):
  * phase-1 (split) kernel time
  * phase-2 (combine) kernel time
  * inter-kernel gap (total - split - combine)
  * end-to-end latency
  * memory-floor anchor: bytes moved by a pure memcpy matching partial_acc bytes
  * NUM_SPLITS=4,8,16 sweeps
  * num_warps sweep on split kernel (4 vs 8)
  * combine kernel standalone (is it mem-bound?)

Uses torch.cuda.Event with flush sync between events for accurate per-kernel
timing. This is *not* used in the benchmark-eval path (that would pick up
event-insertion overhead); it runs offline for investigative purposes only.

Launch:
    modal run scripts/profile_kernel.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("flashinfer-profile")

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
    .env({
        "TORCH_EXTENSIONS_DIR": "/results/torch_ext_cache",
        "DSA_CUTLASS_DIR": "/results/cutlass",
    })
)


# Ship the kernel source inline so Modal bakes it into the container
# independent of whether the user mounted their code.
KERNEL_SRC_PATH = PROJECT_ROOT / "solution" / "triton" / "sparse_fused.py"


@app.function(image=image, gpu="B200:1", timeout=1800, volumes={TRACE_SET_PATH: trace_volume})
def run_profile(kernel_source: str) -> dict:
    import logging
    import statistics
    import math
    import tempfile
    import importlib.util
    import os
    import torch
    import numpy as np
    import triton
    import triton.language as tl
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    # Write the kernel source into a temp file and import it fresh so we can
    # monkey-patch its `kernel()` function with event-instrumentation variants.
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

    # Pick one T=1 and one T=8 workload.
    wls_by_T = {}
    for w in workloads:
        T = int(w.workload.axes.get("num_tokens"))
        wls_by_T.setdefault(T, []).append(w)

    targets = {}
    for T in (1, 8):
        if T in wls_by_T:
            targets[T] = wls_by_T[T][0]
            log.info(f"T={T}: using workload {targets[T].workload.uuid[:8]}")

    def load_inputs(wrapped_wl):
        """Materialize tensors from workload spec."""
        wl = wrapped_wl.workload
        T = int(wl.axes.get("num_tokens"))
        P = int(wl.axes.get("num_pages"))

        # Deterministic seed so tensors are identical across runs.
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
        # sm_scale — latency-irrelevant, any valid scalar works for timing.
        sm_scale = 1.0 / math.sqrt(192)

        out = torch.empty(T, 16, 512, dtype=torch.bfloat16, device=device)
        lse = torch.empty(T, 16, dtype=torch.float32, device=device)
        return q_nope, q_pe, ckv, kpe, sparse_indices, sm_scale, out, lse, T, P

    # Instrumented kernel variants
    def run_kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
                   sm_scale, output, lse, NUM_SPLITS=8, BLOCK_N=64,
                   num_warps_split=8, num_stages_split=2,
                   num_warps_combine=4, num_stages_combine=1,
                   instrument=False):
        num_tokens, H, D_ckv = q_nope.shape
        D_kpe = q_pe.shape[-1]
        TOPK = sparse_indices.shape[-1]
        num_pages, page_size, _ = ckv_cache.shape
        ckv_flat = ckv_cache.view(num_pages * page_size, D_ckv)
        kpe_flat = kpe_cache.view(num_pages * page_size, D_kpe)
        device = q_nope.device
        partial_m = torch.empty((num_tokens, NUM_SPLITS, H), dtype=torch.float32, device=device)
        partial_l = torch.empty((num_tokens, NUM_SPLITS, H), dtype=torch.float32, device=device)
        partial_acc = torch.empty((num_tokens, NUM_SPLITS, H, D_ckv), dtype=torch.float32, device=device)

        events = None
        if instrument:
            events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            events[0].record()

        grid1 = (num_tokens, NUM_SPLITS)
        split_kernel[grid1](
            q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices,
            partial_m, partial_l, partial_acc,
            sm_scale * LOG2E,
            q_nope.stride(0), q_nope.stride(1),
            q_pe.stride(0), q_pe.stride(1),
            ckv_flat.stride(0), kpe_flat.stride(0),
            sparse_indices.stride(0),
            partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
            partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
            partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
            TOPK=TOPK, H=H, D_CKV=D_ckv, D_KPE=D_kpe,
            BLOCK_N=BLOCK_N, NUM_SPLITS=NUM_SPLITS,
            num_warps=num_warps_split, num_stages=num_stages_split,
        )

        if instrument:
            events[1].record()
            events[2].record()

        grid2 = (num_tokens,)
        combine_kernel[grid2](
            partial_m, partial_l, partial_acc, output, lse,
            partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
            partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
            partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
            output.stride(0), output.stride(1),
            lse.stride(0),
            H=H, D_CKV=D_ckv, NUM_SPLITS=NUM_SPLITS,
            num_warps=num_warps_combine, num_stages=num_stages_combine,
        )

        if instrument:
            events[3].record()

        return events, partial_m, partial_l, partial_acc

    def time_total_ms(fn, warmup=10, iters=200):
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
        return s.elapsed_time(e) / iters  # ms

    def phase_breakdown_ms(launcher, warmup=20, trials=200):
        """Return (split_ms, combine_ms, total_ms) medians.

        We measure each trial's own (split, combine, total) with three events
        and take the median separately. With 200 trials the inter-event timer
        quantization (~0.5 us) averages out. We also measure total_ms a second
        time without events to verify the instrumentation doesn't perturb.
        """
        # Warm
        for _ in range(warmup):
            launcher(instrument=False)
        torch.cuda.synchronize()

        split_ms_list = []
        combine_ms_list = []
        total_ms_list = []
        # Many back-to-back trials to overcome event granularity (~0.5 us).
        for _ in range(trials):
            events, *_ = launcher(instrument=True)
            torch.cuda.synchronize()
            split_ms_list.append(events[0].elapsed_time(events[1]))
            combine_ms_list.append(events[2].elapsed_time(events[3]))
            total_ms_list.append(events[0].elapsed_time(events[3]))

        def p(xs, q):
            xs = sorted(xs)
            return xs[int(q * len(xs))]

        return {
            "split_p50_us": p(split_ms_list, 0.5) * 1000,
            "split_p90_us": p(split_ms_list, 0.9) * 1000,
            "combine_p50_us": p(combine_ms_list, 0.5) * 1000,
            "combine_p90_us": p(combine_ms_list, 0.9) * 1000,
            "total_p50_us": p(total_ms_list, 0.5) * 1000,
            "total_p90_us": p(total_ms_list, 0.9) * 1000,
        }

    results = {"configs": {}, "phase": {}, "floor": {}, "split_only": {}, "combine_only": {}}

    for T, wrapped in targets.items():
        q_nope, q_pe, ckv, kpe, si, sm_scale, out, lse, _, P = load_inputs(wrapped)
        log.info(f"T={T}: P={P}, sparse_indices valid-per-token = "
                 f"{(si >= 0).sum(dim=-1).tolist()}")

        # --- 1. Phase breakdown with NUM_SPLITS=8 (the current config)
        def launcher_default(instrument=False):
            return run_kernel(q_nope, q_pe, ckv, kpe, si, sm_scale, out, lse,
                              NUM_SPLITS=8, BLOCK_N=64,
                              num_warps_split=8, num_stages_split=2,
                              num_warps_combine=4, num_stages_combine=1,
                              instrument=instrument)

        # Uninstrumented total (gold-standard).
        total_clean_ms = time_total_ms(lambda: launcher_default(instrument=False))
        log.info(f"T={T}: total (no events) p_avg = {total_clean_ms*1000:.2f} us")

        phases = phase_breakdown_ms(launcher_default)
        phases["total_clean_us"] = total_clean_ms * 1000
        phases["gap_us"] = phases["total_p50_us"] - phases["split_p50_us"] - phases["combine_p50_us"]
        results["phase"][T] = phases
        log.info(f"T={T}: phase={phases}")

        # --- 2. Split-only and combine-only clean timings (back-to-back kernel alone)
        num_tokens = T
        H, D_ckv = 16, 512
        TOPK = 2048
        NUM_SPLITS = 8

        ckv_flat = ckv.view(ckv.shape[0] * ckv.shape[1], ckv.shape[2])
        kpe_flat = kpe.view(kpe.shape[0] * kpe.shape[1], kpe.shape[2])
        partial_m = torch.empty((num_tokens, NUM_SPLITS, H), dtype=torch.float32, device=q_nope.device)
        partial_l = torch.empty((num_tokens, NUM_SPLITS, H), dtype=torch.float32, device=q_nope.device)
        partial_acc = torch.empty((num_tokens, NUM_SPLITS, H, D_ckv), dtype=torch.float32, device=q_nope.device)

        def split_only():
            split_kernel[(num_tokens, NUM_SPLITS)](
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

        def combine_only():
            combine_kernel[(num_tokens,)](
                partial_m, partial_l, partial_acc, out, lse,
                partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
                partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
                partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
                out.stride(0), out.stride(1),
                lse.stride(0),
                H=H, D_CKV=D_ckv, NUM_SPLITS=NUM_SPLITS,
                num_warps=4, num_stages=1,
            )

        # populate partial_* so combine has real inputs
        split_only()
        torch.cuda.synchronize()

        split_alone_us = time_total_ms(split_only) * 1000
        combine_alone_us = time_total_ms(combine_only) * 1000
        results["split_only"][T] = split_alone_us
        results["combine_only"][T] = combine_alone_us
        log.info(f"T={T}: split_only={split_alone_us:.2f} us, combine_only={combine_alone_us:.2f} us")

        # --- 3. Memory-floor anchor.
        # Combine reads NUM_SPLITS * H * D_ckv * 4 bytes + NUM_SPLITS * 2H * 4
        # Writes H * D_ckv * 2 bytes (bf16) + H * 4 bytes (lse)
        # Approximate with a memcpy of the same byte volume.
        bytes_read_combine = (T * NUM_SPLITS * H * D_ckv * 4) + (T * NUM_SPLITS * 2 * H * 4)
        bytes_write_combine = (T * H * D_ckv * 2) + (T * H * 4)
        floor_bytes = bytes_read_combine + bytes_write_combine

        floor_src = torch.empty(floor_bytes // 4, dtype=torch.float32, device=q_nope.device)
        floor_dst = torch.empty_like(floor_src)
        def memcpy_floor():
            floor_dst.copy_(floor_src)
        mem_floor_us = time_total_ms(memcpy_floor) * 1000
        results["floor"][T] = {
            "bytes": floor_bytes,
            "memcpy_us": mem_floor_us,
        }
        log.info(f"T={T}: memory-floor ({floor_bytes/1024:.0f} KB) memcpy={mem_floor_us:.2f} us")

        # --- 4. NUM_SPLITS sweep (4, 8, 16)
        sweep = {}
        for ns in (4, 8, 16):
            def launcher_ns(instrument=False, ns=ns):
                return run_kernel(q_nope, q_pe, ckv, kpe, si, sm_scale, out, lse,
                                  NUM_SPLITS=ns, BLOCK_N=64,
                                  num_warps_split=8, num_stages_split=2,
                                  num_warps_combine=4, num_stages_combine=1,
                                  instrument=instrument)
            total_us = time_total_ms(lambda ln=launcher_ns: ln(instrument=False)) * 1000
            ph = phase_breakdown_ms(launcher_ns, warmup=10, trials=100)
            sweep[ns] = {
                "total_us": total_us,
                "split_p50_us": ph["split_p50_us"],
                "combine_p50_us": ph["combine_p50_us"],
                "gap_us": ph["total_p50_us"] - ph["split_p50_us"] - ph["combine_p50_us"],
            }
            log.info(f"T={T}: NUM_SPLITS={ns}: total={total_us:.2f} us, split={ph['split_p50_us']:.2f}, "
                     f"combine={ph['combine_p50_us']:.2f}")
        results["configs"].setdefault("num_splits_sweep", {})[T] = sweep

        # --- 5. num_warps sweep on split kernel (4 vs 8) with NUM_SPLITS=8
        warp_sweep = {}
        for nw in (4, 8):
            def launcher_nw(instrument=False, nw=nw):
                return run_kernel(q_nope, q_pe, ckv, kpe, si, sm_scale, out, lse,
                                  NUM_SPLITS=8, BLOCK_N=64,
                                  num_warps_split=nw, num_stages_split=2,
                                  num_warps_combine=4, num_stages_combine=1,
                                  instrument=instrument)
            total_us = time_total_ms(lambda ln=launcher_nw: ln(instrument=False)) * 1000
            warp_sweep[nw] = total_us
            log.info(f"T={T}: num_warps_split={nw}: total={total_us:.2f} us")
        results["configs"].setdefault("num_warps_sweep", {})[T] = warp_sweep

        # --- 6. Combine kernel memory pressure: measure combine alone vs
        # a pure zero-compute combine (just load partial_acc and reduce to output
        # without softmax math) — emulate "mem-floor" by replacing the kernel.
        # We approximate by counting bytes and dividing by B200 peak BW (8 TB/s).
        peak_bw_gbs = 8000  # B200 HBM3e
        theo_bw_us = (bytes_read_combine + bytes_write_combine) / (peak_bw_gbs * 1e9) * 1e6
        results["floor"][T]["theoretical_bw_us"] = theo_bw_us

    return results


@app.local_entrypoint()
def main():
    kernel_source = KERNEL_SRC_PATH.read_text()
    print(f"Profiling {KERNEL_SRC_PATH} on Modal B200...")
    out = run_profile.remote(kernel_source)
    import json
    print("\n=== FULL RESULT ===")
    print(json.dumps(out, indent=2, default=str))
