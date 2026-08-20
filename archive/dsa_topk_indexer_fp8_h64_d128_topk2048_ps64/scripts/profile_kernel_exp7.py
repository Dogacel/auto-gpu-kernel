"""
Phase-level profiler for the DSA TopK Indexer kernel at exp 7 (commit a7895bf).

Structure has changed materially since the previous profile:
- exp 6 replaced the ~225 µs .contiguous() K cache copies with zero-copy
  torch.as_strided views.
- exp 7 replaced the 15 torch post-ops (clamp/div/mod/gather/mask/...)
  with a single Triton remap_kernel launch.

So the phases to time now are:
    py_setup    — batch/shape reads, as_strided view construction, scores alloc
    score       — score_kernel launch + execution
    topk        — torch.topk(scores, 2048, dim=-1)
    remap       — remap_kernel (fused post-op Triton kernel)
    total       — end-to-end (python entry → last kernel completes)

We also measure a memory-bandwidth anchor — an unrelated bf16 matmul and
an fp8 memcpy at the same byte volume — for context only.

IMPORTANT: Does NOT modify solution/triton/indexer_fused.py. Uses a host
wrapper that reproduces the shipping `kernel()` body with cuda.Event pairs
between each phase.
"""

import sys
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Patch DNS for sandboxed networks where api.modal.com is not resolvable
# via the usual resolver but is reachable via proxy.
try:
    from scripts._dns_patch import patch as _patch_dns
    _patch_dns()
except Exception:
    pass

import modal

app = modal.App("flashinfer-profile-exp7")

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
    """Profile the current exp-7 kernel, phase-split, on stride-N workloads."""
    import importlib.util
    import logging
    import time
    import torch
    import triton
    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
    from flashinfer_bench.bench.evaluators.utils import allocate_outputs

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    spec = importlib.util.spec_from_file_location("indexer_fused", "/workspace/indexer_fused.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    score_kernel = mod.score_kernel
    remap_kernel = mod.remap_kernel
    kernel_ref = mod.kernel

    logger.info(f"Loading trace set from {TRACE_SET_PATH}")
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition = trace_set.definitions[DEFINITION]
    workload_traces = trace_set.workloads.get(DEFINITION, [])
    workloads = [t.workload for t in workload_traces]
    if stride > 1:
        workloads = workloads[::stride]
    logger.info(f"Profiling {len(workloads)} workload(s) with stride={stride}")

    device = torch.device("cuda")

    def load_inputs(wl):
        safe = load_safetensors(definition, wl, trace_set_root=trace_set.root)
        vals = gen_inputs(definition, wl, device=str(device), safe_tensors=safe)
        names = list(definition.inputs.keys())
        inputs = dict(zip(names, vals))
        out_list = allocate_outputs(definition, vals, device=str(device))
        out_names = list(definition.outputs.keys())
        for name, t in zip(out_names, out_list):
            inputs[name] = t
        return inputs

    # Phased host wrapper mirroring solution/triton/indexer_fused.py::kernel
    # exactly (no logic change, only cuda.Event pairs added between phases).
    def timed(inputs, n_iters):
        q_index_fp8 = inputs["q_index_fp8"]
        k_index_cache_fp8 = inputs["k_index_cache_fp8"]
        weights = inputs["weights"]
        seq_lens = inputs["seq_lens"]
        block_table = inputs["block_table"]
        topk_indices = inputs["topk_indices"]

        per_iter = {k: [] for k in (
            "py_setup_us", "score_us", "topk_us", "remap_us",
            "total_us", "cpu_total_us",
        )}

        # Warm up Triton JIT cache using the real kernel (no events).
        for _ in range(5):
            kernel_ref(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices)
        torch.cuda.synchronize()

        for _ in range(n_iters):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e2 = torch.cuda.Event(enable_timing=True)
            e3 = torch.cuda.Event(enable_timing=True)
            e4 = torch.cuda.Event(enable_timing=True)

            t0 = time.perf_counter_ns()
            e0.record()

            # Phase py_setup — mirrors the exp-7 kernel prologue verbatim:
            # shape reads, as_strided view construction, scores allocation.
            batch_size, H, D = q_index_fp8.shape
            num_pages, page_size, _, head_dim_sf = k_index_cache_fp8.shape
            head_dim = head_dim_sf - 4
            _, max_num_pages = block_table.shape
            topk = 2048

            page_bytes = page_size * head_dim_sf
            fp8_view = torch.as_strided(
                k_index_cache_fp8.view(torch.float8_e4m3fn),
                size=(num_pages, page_size, head_dim),
                stride=(page_bytes, head_dim, 1),
            )
            scale_view = torch.as_strided(
                k_index_cache_fp8.view(torch.float32),
                size=(num_pages, page_size),
                stride=(page_bytes // 4, 1),
                storage_offset=page_size * head_dim // 4,
            )

            max_scored = max_num_pages * page_size
            scores = torch.empty(
                (batch_size, max_scored), device=q_index_fp8.device, dtype=torch.float32
            )
            e1.record()

            # Phase score — Triton score_kernel.
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

            # Phase topk — torch.topk.
            effective_topk = min(topk, max_scored)
            _, topk_idx = torch.topk(scores, effective_topk, dim=-1)
            e3.record()

            # Phase remap — Triton remap_kernel.
            BLOCK_K = 256
            remap_grid = (batch_size, triton.cdiv(topk, BLOCK_K))
            remap_kernel[remap_grid](
                topk_idx, block_table, seq_lens, topk_indices,
                topk_idx.stride(0), topk_idx.stride(1),
                block_table.stride(0), block_table.stride(1),
                topk_indices.stride(0), topk_indices.stride(1),
                page_size=page_size,
                max_num_pages=max_num_pages,
                effective_topk=effective_topk,
                topk=topk,
                BLOCK_K=BLOCK_K,
            )
            e4.record()

            t1 = time.perf_counter_ns()
            torch.cuda.synchronize()

            per_iter["py_setup_us"].append(e0.elapsed_time(e1) * 1000.0)
            per_iter["score_us"].append(e1.elapsed_time(e2) * 1000.0)
            per_iter["topk_us"].append(e2.elapsed_time(e3) * 1000.0)
            per_iter["remap_us"].append(e3.elapsed_time(e4) * 1000.0)
            per_iter["total_us"].append(e0.elapsed_time(e4) * 1000.0)
            per_iter["cpu_total_us"].append((t1 - t0) / 1000.0)

        return per_iter

    def timed_no_alloc(inputs, n_iters):
        """Isolates the score_kernel+topk+remap cost if the `scores` buffer
        were preallocated (hypothetical hoist).

        Hints at whether torch.empty is a meaningful fraction of py_setup.
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
        max_scored = max_num_pages * page_size
        effective_topk = min(topk, max_scored)

        page_bytes = page_size * head_dim_sf
        fp8_view = torch.as_strided(
            k_index_cache_fp8.view(torch.float8_e4m3fn),
            size=(num_pages, page_size, head_dim),
            stride=(page_bytes, head_dim, 1),
        )
        scale_view = torch.as_strided(
            k_index_cache_fp8.view(torch.float32),
            size=(num_pages, page_size),
            stride=(page_bytes // 4, 1),
            storage_offset=page_size * head_dim // 4,
        )
        scores = torch.empty(
            (batch_size, max_scored), device=q_index_fp8.device, dtype=torch.float32
        )

        for _ in range(5):
            kernel_ref(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices)
        torch.cuda.synchronize()

        times_us = []
        for _ in range(n_iters):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
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
            _, topk_idx = torch.topk(scores, effective_topk, dim=-1)
            BLOCK_K = 256
            remap_grid = (batch_size, triton.cdiv(topk, BLOCK_K))
            remap_kernel[remap_grid](
                topk_idx, block_table, seq_lens, topk_indices,
                topk_idx.stride(0), topk_idx.stride(1),
                block_table.stride(0), block_table.stride(1),
                topk_indices.stride(0), topk_indices.stride(1),
                page_size=page_size,
                max_num_pages=max_num_pages,
                effective_topk=effective_topk,
                topk=topk,
                BLOCK_K=BLOCK_K,
            )
            e1.record()
            torch.cuda.synchronize()
            times_us.append(e0.elapsed_time(e1) * 1000.0)
        return times_us

    def stub_topk(inputs, n_iters):
        """Back-out a lower bound on non-topk latency by replacing
        torch.topk with a cheap stub that produces the same-shaped int64
        output. Difference versus `timed()` attributes a cost to topk.

        We use torch.arange().expand() to produce [B, topk] of zeros;
        downstream remap treats them as valid indices (it's fine for
        timing — we only care about kernel dispatch cost, not output).
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
        max_scored = max_num_pages * page_size
        effective_topk = min(topk, max_scored)

        page_bytes = page_size * head_dim_sf
        fp8_view = torch.as_strided(
            k_index_cache_fp8.view(torch.float8_e4m3fn),
            size=(num_pages, page_size, head_dim),
            stride=(page_bytes, head_dim, 1),
        )
        scale_view = torch.as_strided(
            k_index_cache_fp8.view(torch.float32),
            size=(num_pages, page_size),
            stride=(page_bytes // 4, 1),
            storage_offset=page_size * head_dim // 4,
        )

        topk_idx_stub = torch.zeros((batch_size, effective_topk), device=device, dtype=torch.int64)

        for _ in range(5):
            kernel_ref(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices)
        torch.cuda.synchronize()

        times_us = []
        for _ in range(n_iters):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
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
            # Stub: do NOT call torch.topk. Use a pre-allocated dummy.
            topk_idx = topk_idx_stub
            BLOCK_K = 256
            remap_grid = (batch_size, triton.cdiv(topk, BLOCK_K))
            remap_kernel[remap_grid](
                topk_idx, block_table, seq_lens, topk_indices,
                topk_idx.stride(0), topk_idx.stride(1),
                block_table.stride(0), block_table.stride(1),
                topk_indices.stride(0), topk_indices.stride(1),
                page_size=page_size,
                max_num_pages=max_num_pages,
                effective_topk=effective_topk,
                topk=topk,
                BLOCK_K=BLOCK_K,
            )
            e1.record()
            torch.cuda.synchronize()
            times_us.append(e0.elapsed_time(e1) * 1000.0)
        return times_us

    def mem_floor(inputs, n_iters=30):
        """Pure K cache memcpy as a memory-bandwidth anchor at the kernel's byte volume."""
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
        num_prog = B * max_num_pages

        try:
            per_iter = timed(inputs, n_iters=iters)
        except Exception as e:
            logger.exception(f"Workload {wl.uuid[:8]} timed failed: {e}")
            continue

        try:
            no_alloc_us = timed_no_alloc(inputs, n_iters=min(iters, 30))
        except Exception as e:
            logger.warning(f"Workload {wl.uuid[:8]} no_alloc failed: {e}")
            no_alloc_us = []

        try:
            stub_topk_us = stub_topk(inputs, n_iters=min(iters, 30))
        except Exception as e:
            logger.warning(f"Workload {wl.uuid[:8]} stub_topk failed: {e}")
            stub_topk_us = []

        try:
            mem_us = mem_floor(inputs)
        except Exception as e:
            logger.warning(f"Workload {wl.uuid[:8]} mem_floor failed: {e}")
            mem_us = []

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
        for phase in ("py_setup_us", "score_us", "topk_us", "remap_us", "total_us", "cpu_total_us"):
            xs = per_iter[phase]
            result[f"{phase}_min"] = min(xs)
            result[f"{phase}_p50"] = pct(xs, 50)
            result[f"{phase}_p90"] = pct(xs, 90)
            result[f"{phase}_mean"] = statistics.mean(xs)
        result["no_alloc_p50_us"] = pct(no_alloc_us, 50) if no_alloc_us else 0.0
        result["no_alloc_p90_us"] = pct(no_alloc_us, 90) if no_alloc_us else 0.0
        result["stub_topk_p50_us"] = pct(stub_topk_us, 50) if stub_topk_us else 0.0
        result["stub_topk_p90_us"] = pct(stub_topk_us, 90) if stub_topk_us else 0.0
        result["mem_floor_p50_us"] = pct(mem_us, 50) if mem_us else 0.0

        results.append(result)
        logger.info(
            f"[{i+1}/{len(workloads)}] {wl.uuid[:8]} B={B} pg={max_num_pages} "
            f"prog={num_prog} sum_sl={sum_sl} | "
            f"tot={result['total_us_p50']:5.1f} "
            f"py={result['py_setup_us_p50']:5.1f} "
            f"score={result['score_us_p50']:5.1f} "
            f"topk={result['topk_us_p50']:5.1f} "
            f"remap={result['remap_us_p50']:5.1f} "
            f"| no_alloc={result['no_alloc_p50_us']:5.1f} "
            f"stub_topk={result['stub_topk_p50_us']:5.1f} "
            f"mem_floor={result['mem_floor_p50_us']:5.1f}"
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
        ("py_setup_us_p50", "py_setup"),
        ("score_us_p50", "score_kernel"),
        ("topk_us_p50", "topk"),
        ("remap_us_p50", "remap_kernel"),
        ("total_us_p50", "TOTAL"),
        ("cpu_total_us_p50", "cpu_total"),
        ("no_alloc_p50_us", "no_alloc_total"),
        ("stub_topk_p50_us", "stub_topk_total"),
    ]:
        sp = agg(small, phase_key)
        lp = agg(large, phase_key)
        print(f"{label:<16} {sp[0]:>12.2f} {sp[1]:>12.2f} {lp[0]:>12.2f} {lp[1]:>12.2f}")

    print(f"\nMemory floor anchor: small p50={agg(small, 'mem_floor_p50_us')[0]:.2f} µs, large p50={agg(large, 'mem_floor_p50_us')[0]:.2f} µs")

    worst_tot = sorted(results, key=lambda r: -r["total_us_p50"])[:5]
    print(f"\nWorst-5 by total µs (p50):")
    print(f"  {'uuid':10} {'B':>3} {'pg':>4} {'prog':>5} {'sum_sl':>7} "
          f"{'tot':>7} {'py':>7} {'score':>7} {'topk':>7} {'remap':>7}")
    for r in worst_tot:
        print(f"  {r['uuid'][:8]:10} {r['B']:>3} {r['max_num_pages']:>4} "
              f"{r['num_programs']:>5} {r['sum_seq_len']:>7} "
              f"{r['total_us_p50']:>7.1f} {r['py_setup_us_p50']:>7.1f} "
              f"{r['score_us_p50']:>7.1f} {r['topk_us_p50']:>7.1f} "
              f"{r['remap_us_p50']:>7.1f}")

    worst_pt = sorted(results, key=lambda r: -(r["total_us_p50"] / max(r["sum_seq_len"], 1)))[:5]
    print(f"\nWorst-5 by µs per sum(seq_len):")
    for r in worst_pt:
        pt = r["total_us_p50"] / max(r["sum_seq_len"], 1)
        print(f"  {r['uuid'][:8]:10} B={r['B']:>3} sum_sl={r['sum_seq_len']:>7} "
              f"| µs/tok={pt:.4f} total={r['total_us_p50']:.1f}")

    import json
    out_path = PROJECT_ROOT / "experiments" / "profile_exp7_raw.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results → {out_path}")
