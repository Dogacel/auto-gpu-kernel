"""
Phase-level profiler for the DSA TopK Indexer kernel.

Runs the current kernel on the Modal trace workloads, but splits the
pipeline into phases via torch.cuda.Event pairs so we can see where
the ~0.31 ms mean latency goes.

Phases timed:
    setup       — FP8 SOA extract (.contiguous()), buffer alloc
    score       — Triton score_kernel launch + execution
    topk        — torch.topk(scores, 2048)
    remap       — clamp/mod/gather/mul+add (page-remap arithmetic)
    mask_write  — arange/where/fill_/copy_ (masked write)
    total       — full end-to-end

Also runs a memory-bandwidth anchor (bf16 matmul + pure fp8 .contiguous()
copy of the K cache) for comparison with `score` µs to see if
score_kernel is memory-bound.

IMPORTANT: Does NOT modify solution/triton/indexer_fused.py.
"""

import sys
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Patch DNS for sandboxed networks where api.modal.com is not resolvable
# via the usual resolver but is reachable via proxy. Only relevant
# locally; inside the Modal container the module won't exist and we
# simply skip.
try:
    from scripts._dns_patch import patch as _patch_dns
    _patch_dns()
except Exception:
    pass

import modal

app = modal.App("flashinfer-profile")

TRACE_SET_PATH = "/data"
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)

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
    .add_local_file(
        str(PROJECT_ROOT / "solution" / "triton" / "indexer_fused.py"),
        "/workspace/indexer_fused.py",
        copy=True,
    )
)


DEFINITION = "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64"


@app.function(
    image=image,
    gpu="B200:1",
    timeout=2400,
    retries=0,
    volumes={TRACE_SET_PATH: trace_volume},
)
def run_profile(stride: int = 8, iters: int = 50) -> dict:
    """Profile the current kernel, phase-split, on stride-N workloads."""
    import importlib.util
    import logging
    import torch
    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
    from flashinfer_bench.bench.evaluators.utils import allocate_outputs

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    spec = importlib.util.spec_from_file_location("indexer_fused", "/workspace/indexer_fused.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    score_kernel = mod.score_kernel
    kernel_ref = mod.kernel

    logger.info(f"Loading trace set from {TRACE_SET_PATH}")
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition = trace_set.definitions[DEFINITION]
    # TraceSet.workloads stores Trace objects — unwrap to their .workload.
    workload_traces = trace_set.workloads.get(DEFINITION, [])
    workloads = [t.workload for t in workload_traces]
    if stride > 1:
        workloads = workloads[::stride]
    logger.info(f"Profiling {len(workloads)} workload(s) with stride={stride}")

    device = torch.device("cuda")

    def load_inputs(wl):
        """Materialize workload tensors onto GPU in definition-input order.

        Also allocates definition.outputs (destination-passing-style) since the
        kernel signature expects them as positional args after the inputs."""
        safe = load_safetensors(definition, wl, trace_set_root=trace_set.root)
        vals = gen_inputs(definition, wl, device=str(device), safe_tensors=safe)
        names = list(definition.inputs.keys())
        inputs = dict(zip(names, vals))
        out_list = allocate_outputs(definition, vals, device=str(device))
        out_names = list(definition.outputs.keys())
        for name, t in zip(out_names, out_list):
            inputs[name] = t
        return inputs

    # Phased host wrapper mirroring solution/triton/indexer_fused.py::kernel.
    def timed(inputs, n_iters):
        q_index_fp8 = inputs["q_index_fp8"]
        k_index_cache_fp8 = inputs["k_index_cache_fp8"]
        weights = inputs["weights"]
        seq_lens = inputs["seq_lens"]
        block_table = inputs["block_table"]
        topk_indices = inputs["topk_indices"]

        batch_size, H, D = q_index_fp8.shape
        num_pages, page_size, _, head_dim_sf = k_index_cache_fp8.shape
        head_dim = head_dim_sf - 4
        _, max_num_pages = block_table.shape
        topk = 2048
        kv_u8 = k_index_cache_fp8.view(torch.uint8).reshape(num_pages, page_size * head_dim_sf)

        per_iter = {k: [] for k in (
            "setup_us", "fp8copy_us", "scalecopy_us", "alloc_us",
            "score_us", "topk_us", "remap_us", "mask_us", "total_us",
        )}

        # Warmup using the real (un-instrumented) kernel so Triton caches
        # are warm and the kernel's own compilation is excluded.
        for _ in range(5):
            kernel_ref(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices)
        torch.cuda.synchronize()

        for _ in range(n_iters):
            e0 = torch.cuda.Event(enable_timing=True)
            e_fp8 = torch.cuda.Event(enable_timing=True)
            e_scale = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)  # after alloc
            e2 = torch.cuda.Event(enable_timing=True)
            e3 = torch.cuda.Event(enable_timing=True)
            e4 = torch.cuda.Event(enable_timing=True)
            e5 = torch.cuda.Event(enable_timing=True)

            e0.record()

            # Phase 0a: fp8 slice .contiguous() copy (full K cache)
            fp8_view = (
                kv_u8[:, :page_size * head_dim]
                .contiguous()
                .view(num_pages, page_size, head_dim)
                .view(torch.float8_e4m3fn)
            )
            e_fp8.record()

            # Phase 0b: scale slice .contiguous() copy
            scale_view = (
                kv_u8[:, page_size * head_dim:]
                .contiguous()
                .view(num_pages, page_size, 4)
                .view(torch.float32)
                .squeeze(-1)
            )
            e_scale.record()

            # Phase 0c: allocate scores buffer
            max_scored = max_num_pages * page_size
            scores = torch.empty(
                (batch_size, max_scored), device=q_index_fp8.device, dtype=torch.float32
            )
            e1.record()

            # Phase 1: score kernel
            grid = (batch_size, max_num_pages)
            score_kernel[grid](
                q_index_fp8, fp8_view, scale_view, weights, seq_lens, block_table, scores,
                q_index_fp8.stride(0), q_index_fp8.stride(1), q_index_fp8.stride(2),
                fp8_view.stride(0), fp8_view.stride(1), fp8_view.stride(2),
                scale_view.stride(0), scale_view.stride(1),
                weights.stride(0), weights.stride(1),
                block_table.stride(0), block_table.stride(1),
                scores.stride(0), scores.stride(1),
                BLOCK_H=H, BLOCK_D=D, BLOCK_T=page_size,
            )
            e2.record()

            # Phase 2: topk
            effective_topk = min(topk, max_scored)
            _, topk_idx = torch.topk(scores, effective_topk, dim=-1)
            e3.record()

            # Phase 3: remap
            page_idx_per_token = (topk_idx // page_size).clamp_(max=max_num_pages - 1)
            offset_per_token = topk_idx % page_size
            bt_long = block_table.to(torch.long)
            global_page_idx = torch.gather(bt_long, 1, page_idx_per_token)
            topk_tokens = (global_page_idx * page_size + offset_per_token).to(torch.int32)
            e4.record()

            # Phase 4: mask_write
            seq_lens_long = seq_lens.to(torch.long)
            actual_topks = torch.minimum(
                seq_lens_long, torch.full_like(seq_lens_long, effective_topk)
            )
            arange = torch.arange(effective_topk, device=device).unsqueeze(0)
            mask = arange < actual_topks.unsqueeze(-1)
            masked = torch.where(mask, topk_tokens, torch.full_like(topk_tokens, -1))
            topk_indices.fill_(-1)
            topk_indices[:, :effective_topk].copy_(masked)
            e5.record()

            torch.cuda.synchronize()

            per_iter["setup_us"].append(e0.elapsed_time(e1) * 1000.0)
            per_iter["fp8copy_us"].append(e0.elapsed_time(e_fp8) * 1000.0)
            per_iter["scalecopy_us"].append(e_fp8.elapsed_time(e_scale) * 1000.0)
            per_iter["alloc_us"].append(e_scale.elapsed_time(e1) * 1000.0)
            per_iter["score_us"].append(e1.elapsed_time(e2) * 1000.0)
            per_iter["topk_us"].append(e2.elapsed_time(e3) * 1000.0)
            per_iter["remap_us"].append(e3.elapsed_time(e4) * 1000.0)
            per_iter["mask_us"].append(e4.elapsed_time(e5) * 1000.0)
            per_iter["total_us"].append(e0.elapsed_time(e5) * 1000.0)

        return per_iter

    def timed_no_setup(inputs, n_iters):
        """Same as timed() but fp8_view/scale_view precomputed once.

        Isolates the cost you would save by hoisting the .contiguous() copies
        out of the per-call path (e.g., caching them across calls or fusing
        dequant into the kernel).
        """
        q_index_fp8 = inputs["q_index_fp8"]
        k_index_cache_fp8 = inputs["k_index_cache_fp8"]
        weights = inputs["weights"]
        seq_lens = inputs["seq_lens"]
        block_table = inputs["block_table"]
        topk_indices = inputs["topk_indices"]

        batch_size, H, D = q_index_fp8.shape
        num_pages, page_size, _, head_dim_sf = k_index_cache_fp8.shape
        head_dim = head_dim_sf - 4
        _, max_num_pages = block_table.shape
        topk = 2048
        kv_u8 = k_index_cache_fp8.view(torch.uint8).reshape(num_pages, page_size * head_dim_sf)

        # Precompute ONCE: fp8 view + scale view
        fp8_view = (kv_u8[:, :page_size * head_dim].contiguous()
                    .view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn))
        scale_view = (kv_u8[:, page_size * head_dim:].contiguous()
                      .view(num_pages, page_size, 4).view(torch.float32).squeeze(-1))

        # Warmup
        for _ in range(5):
            kernel_ref(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices)
        torch.cuda.synchronize()

        times = []
        for _ in range(n_iters):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()

            max_scored = max_num_pages * page_size
            scores = torch.empty(
                (batch_size, max_scored), device=q_index_fp8.device, dtype=torch.float32
            )
            grid = (batch_size, max_num_pages)
            score_kernel[grid](
                q_index_fp8, fp8_view, scale_view, weights, seq_lens, block_table, scores,
                q_index_fp8.stride(0), q_index_fp8.stride(1), q_index_fp8.stride(2),
                fp8_view.stride(0), fp8_view.stride(1), fp8_view.stride(2),
                scale_view.stride(0), scale_view.stride(1),
                weights.stride(0), weights.stride(1),
                block_table.stride(0), block_table.stride(1),
                scores.stride(0), scores.stride(1),
                BLOCK_H=H, BLOCK_D=D, BLOCK_T=page_size,
            )
            effective_topk = min(topk, max_scored)
            _, topk_idx = torch.topk(scores, effective_topk, dim=-1)
            page_idx_per_token = (topk_idx // page_size).clamp_(max=max_num_pages - 1)
            offset_per_token = topk_idx % page_size
            bt_long = block_table.to(torch.long)
            global_page_idx = torch.gather(bt_long, 1, page_idx_per_token)
            topk_tokens = (global_page_idx * page_size + offset_per_token).to(torch.int32)
            seq_lens_long = seq_lens.to(torch.long)
            actual_topks = torch.minimum(
                seq_lens_long, torch.full_like(seq_lens_long, effective_topk)
            )
            arange = torch.arange(effective_topk, device=device).unsqueeze(0)
            mask = arange < actual_topks.unsqueeze(-1)
            masked = torch.where(mask, topk_tokens, torch.full_like(topk_tokens, -1))
            topk_indices.fill_(-1)
            topk_indices[:, :effective_topk].copy_(masked)
            e1.record()
            torch.cuda.synchronize()
            times.append(e0.elapsed_time(e1) * 1000.0)
        return times

    def mem_floor(inputs, n_iters=30):
        """Anchor 1: pure K cache memcpy (contiguous copy).

        Same exact byte volume as the score kernel's K loads.
        """
        k_index_cache_fp8 = inputs["k_index_cache_fp8"]
        num_pages, page_size, _, head_dim_sf = k_index_cache_fp8.shape
        head_dim = head_dim_sf - 4

        kv_u8 = k_index_cache_fp8.view(torch.uint8).reshape(num_pages, page_size * head_dim_sf)
        fp8_bytes = kv_u8[:, :page_size * head_dim]

        for _ in range(5):
            _ = fp8_bytes.contiguous()
        torch.cuda.synchronize()

        times_us = []
        for _ in range(n_iters):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            _ = fp8_bytes.contiguous()
            e1.record()
            torch.cuda.synchronize()
            times_us.append(e0.elapsed_time(e1) * 1000.0)
        return times_us

    def matmul_floor(inputs, n_iters=30):
        """Anchor 2: bf16 matmul at the same seq-len scale.

        Gives a compute-bound floor, not memory-bound — it's the lowest
        possible latency the MMA pipeline can hit.
        """
        q_index_fp8 = inputs["q_index_fp8"]
        k_index_cache_fp8 = inputs["k_index_cache_fp8"]

        batch_size, H, D = q_index_fp8.shape
        num_pages, page_size, _, head_dim_sf = k_index_cache_fp8.shape
        T = num_pages * page_size

        q_bf16 = torch.randn(batch_size, H, D, device=device, dtype=torch.bfloat16)
        k_bf16 = torch.randn(T, D, device=device, dtype=torch.bfloat16)

        for _ in range(5):
            _ = torch.einsum("bhd,td->bht", q_bf16, k_bf16)
        torch.cuda.synchronize()

        times_us = []
        for _ in range(n_iters):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            _ = torch.einsum("bhd,td->bht", q_bf16, k_bf16)
            e1.record()
            torch.cuda.synchronize()
            times_us.append(e0.elapsed_time(e1) * 1000.0)
        return times_us

    def pct(xs, q):
        if not xs:
            return 0.0
        xs = sorted(xs)
        k = int(round((q / 100) * (len(xs) - 1)))
        return xs[k]

    results = []
    for i, wl in enumerate(workloads):
        try:
            inputs = load_inputs(wl)
        except Exception as e:
            logger.exception(f"Workload {wl.uuid[:8]} load failed: {e}")
            continue

        q_index_fp8 = inputs["q_index_fp8"]
        k_index_cache_fp8 = inputs["k_index_cache_fp8"]
        seq_lens = inputs["seq_lens"]
        block_table = inputs["block_table"]

        B = q_index_fp8.shape[0]
        P, page_size, _, _ = k_index_cache_fp8.shape
        _, max_num_pages = block_table.shape
        max_sl = int(seq_lens.max().item())
        sum_sl = int(seq_lens.sum().item())
        num_prog = B * max_num_pages  # score kernel launches this many programs

        try:
            per_iter = timed(inputs, n_iters=iters)
        except Exception as e:
            logger.exception(f"Workload {wl.uuid[:8]} timed failed: {e}")
            continue

        try:
            mem_us = mem_floor(inputs)
            mm_us = matmul_floor(inputs)
        except Exception as e:
            logger.warning(f"Workload {wl.uuid[:8]} floor failed: {e}")
            mem_us, mm_us = [], []

        try:
            no_setup_us = timed_no_setup(inputs, n_iters=min(iters, 30))
        except Exception as e:
            logger.warning(f"Workload {wl.uuid[:8]} no_setup failed: {e}")
            no_setup_us = []

        result = {
            "uuid": wl.uuid,
            "B": B,
            "max_num_pages": max_num_pages,
            "num_programs": num_prog,
            "num_pages_in_cache": P,
            "max_seq_len": max_sl,
            "sum_seq_len": sum_sl,
            "iters": len(per_iter["total_us"]),
        }
        for phase in ("setup_us", "fp8copy_us", "scalecopy_us", "alloc_us",
                      "score_us", "topk_us", "remap_us", "mask_us", "total_us"):
            xs = per_iter[phase]
            result[f"{phase}_min"] = min(xs)
            result[f"{phase}_p50"] = pct(xs, 50)
            result[f"{phase}_p90"] = pct(xs, 90)
            result[f"{phase}_mean"] = statistics.mean(xs)
        result["mem_floor_p50_us"] = pct(mem_us, 50)
        result["matmul_floor_p50_us"] = pct(mm_us, 50)
        result["no_setup_p50_us"] = pct(no_setup_us, 50) if no_setup_us else 0.0
        result["no_setup_p90_us"] = pct(no_setup_us, 90) if no_setup_us else 0.0

        results.append(result)
        logger.info(
            f"[{i+1}/{len(workloads)}] {wl.uuid[:8]} B={B} pg={max_num_pages} "
            f"prog={num_prog} sum_sl={sum_sl} | "
            f"tot={result['total_us_p50']:5.1f} (fp8={result['fp8copy_us_p50']:5.1f} "
            f"sc={result['scalecopy_us_p50']:4.1f} alloc={result['alloc_us_p50']:4.1f}) "
            f"score={result['score_us_p50']:5.1f} topk={result['topk_us_p50']:5.1f} "
            f"remap={result['remap_us_p50']:5.1f} mask={result['mask_us_p50']:5.1f} "
            f"| NO_SETUP={result['no_setup_p50_us']:5.1f} "
            f"mm_floor={result['matmul_floor_p50_us']:5.1f}"
        )

    return {"results": results}


@app.local_entrypoint()
def main(stride: int = 8, iters: int = 50):
    print(f"Profiling stride={stride} iters={iters}")
    out = run_profile.remote(stride=stride, iters=iters)
    results = out["results"]
    if not results:
        print("No results.")
        return

    # Split small vs large by total µs p50.
    totals = sorted(r["total_us_p50"] for r in results)
    median_total = totals[len(totals) // 2]
    small = [r for r in results if r["total_us_p50"] <= median_total]
    large = [r for r in results if r["total_us_p50"] > median_total]

    def agg(rs, key):
        xs = [r[key] for r in rs]
        if not xs:
            return (0, 0, 0)
        xs_sorted = sorted(xs)
        p50 = xs_sorted[len(xs_sorted) // 2]
        p90 = xs_sorted[int(round(0.9 * (len(xs_sorted) - 1)))]
        return (p50, p90, statistics.mean(xs))

    print(f"\nResults (n={len(results)}, split at total p50 µs={median_total:.1f}):")
    print(f"  small (n={len(small)}): up to {median_total:.1f} µs total")
    print(f"  large (n={len(large)}): above {median_total:.1f} µs total")

    print(f"\n{'Phase':<16} {'small p50':>12} {'small p90':>12} {'large p50':>12} {'large p90':>12}")
    for phase_key, label in [
        ("setup_us_p50", "setup"),
        ("score_us_p50", "score_kernel"),
        ("topk_us_p50", "topk"),
        ("remap_us_p50", "remap"),
        ("mask_us_p50", "mask_write"),
        ("total_us_p50", "TOTAL"),
    ]:
        sp = agg(small, phase_key)
        lp = agg(large, phase_key)
        print(f"{label:<16} {sp[0]:>12.2f} {sp[1]:>12.2f} {lp[0]:>12.2f} {lp[1]:>12.2f}")

    print(f"\nMemory / compute floor:")
    print(f"  mem_floor fp8 copy (K cache): small p50={agg(small, 'mem_floor_p50_us')[0]:.2f} µs, large p50={agg(large, 'mem_floor_p50_us')[0]:.2f} µs")
    print(f"  matmul floor (bf16 einsum)  : small p50={agg(small, 'matmul_floor_p50_us')[0]:.2f} µs, large p50={agg(large, 'matmul_floor_p50_us')[0]:.2f} µs")

    # Worst-5 by total µs
    worst_tot = sorted(results, key=lambda r: -r["total_us_p50"])[:5]
    print(f"\nWorst-5 by total µs (p50):")
    print(f"  {'uuid':10} {'B':>3} {'pg':>4} {'prog':>5} {'sum_sl':>7} {'tot':>7} {'setup':>7} {'score':>7} {'topk':>7} {'remap':>7} {'mask':>7}")
    for r in worst_tot:
        print(f"  {r['uuid'][:8]:10} {r['B']:>3} {r['max_num_pages']:>4} "
              f"{r['num_programs']:>5} {r['sum_seq_len']:>7} "
              f"{r['total_us_p50']:>7.1f} {r['setup_us_p50']:>7.1f} "
              f"{r['score_us_p50']:>7.1f} {r['topk_us_p50']:>7.1f} "
              f"{r['remap_us_p50']:>7.1f} {r['mask_us_p50']:>7.1f}")

    # Worst-5 by µs per token
    worst_pt = sorted(results, key=lambda r: -(r["total_us_p50"] / max(r["sum_seq_len"], 1)))[:5]
    print(f"\nWorst-5 by µs per sum(seq_len):")
    for r in worst_pt:
        pt = r["total_us_p50"] / max(r["sum_seq_len"], 1)
        print(f"  {r['uuid'][:8]:10} B={r['B']:>3} sum_sl={r['sum_seq_len']:>7} "
              f"| µs/tok={pt:.4f} total={r['total_us_p50']:.1f}")

    import json
    out_path = PROJECT_ROOT / "experiments" / "profile_raw.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results → {out_path}")
