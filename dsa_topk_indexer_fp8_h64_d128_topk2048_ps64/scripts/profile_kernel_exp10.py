"""
Phase-level profiler for the DSA TopK Indexer kernel at exp 10 state.

Post exp-9 (early-return) and exp-10 (scale-after-sum), and with five
consecutive reverts (exp 11-15) behind us. Stale exp-7 profile lives
at experiments/profile.md.

Phases timed per iteration:
    py_setup        — shape reads + as_strided views (NO scores alloc)
    alloc           — torch.empty((B, max_scored), dtype=fp32) for scores
    score_kernel    — Triton score_kernel launch + execution
    topk            — torch.topk(scores, effective_topk, dim=-1)
    remap_kernel    — Triton remap_kernel
    total           — py_setup..remap_kernel end

We split py_setup and alloc with a dedicated event pair so we can see
the torch.empty cost by itself (profile-7 lumped them into one).

We also measure:
    cpu_total       — CPU wall-clock over the whole python entry
                      (event_total + cpu_total delta = unidentified gaps
                       = launch overhead + Python dispatch)
    no_alloc_total  — ceiling after preallocating scores buffer
    stub_topk_total — ceiling after replacing torch.topk with a stub

We focus on three hand-picked workloads the user requested:
    small:  30cecff1 (B=1, 2 tokens)
    medium: dba1e960 (to be resolved from the trace set)
    large:  a876010b (B=29, 89 pages)
and additionally pick a few neighbors to back up each regime.

IMPORTANT: Does NOT modify solution/triton/indexer_fused.py. Uses a host
wrapper that reproduces the shipping kernel() body with cuda.Event pairs.
"""

import sys
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts._dns_patch import patch as _patch_dns
    _patch_dns()
except Exception:
    pass

import modal

app = modal.App("flashinfer-profile-exp10")

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

# Hand-picked representative workloads the user specified.
# We profile these with extra iterations and higher-priority reporting.
TARGET_UUIDS = {
    "small":  "30cecff1",
    "medium": "dba1e960",
    "large":  "a876010b",
}


@app.function(
    image=image,
    gpu="B200:1",
    timeout=2400,
    retries=0,
    volumes={TRACE_SET_PATH: trace_volume},
)
def run_profile(iters: int = 100, extra_stride: int = 16) -> dict:
    """Profile the exp-10 kernel. Focuses on 3 target workloads + stride-16 context."""
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
    all_workloads = [t.workload for t in workload_traces]

    # Select workloads: hand-picked targets + stride-16 context.
    picked_uuids = set()
    selected = []

    # 1. Hand-picked targets (high-iteration runs).
    for label, prefix in TARGET_UUIDS.items():
        for wl in all_workloads:
            if wl.uuid.startswith(prefix):
                selected.append((label, wl))
                picked_uuids.add(wl.uuid)
                logger.info(f"target[{label}]: {wl.uuid}")
                break
        else:
            logger.warning(f"target[{label}] prefix={prefix} not found")

    # 2. Add stride-16 context (not hand-picked ones).
    for wl in all_workloads[::extra_stride]:
        if wl.uuid not in picked_uuids:
            selected.append(("context", wl))
            picked_uuids.add(wl.uuid)

    logger.info(f"Profiling {len(selected)} workloads ({len(TARGET_UUIDS)} targets + context)")

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

    # Mirrors solution/triton/indexer_fused.py::kernel exactly (no logic change).
    # The only instrumentation is cuda.Event pairs between each phase.
    def timed(inputs, n_iters):
        q_index_fp8 = inputs["q_index_fp8"]
        k_index_cache_fp8 = inputs["k_index_cache_fp8"]
        weights = inputs["weights"]
        seq_lens = inputs["seq_lens"]
        block_table = inputs["block_table"]
        topk_indices = inputs["topk_indices"]

        per_iter = {k: [] for k in (
            "py_setup_us", "alloc_us", "score_us", "topk_us", "remap_us",
            "total_us", "cpu_total_us",
        )}

        for _ in range(10):
            kernel_ref(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices)
        torch.cuda.synchronize()

        for _ in range(n_iters):
            e0 = torch.cuda.Event(enable_timing=True)
            e_alloc_start = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)  # after alloc, before score
            e2 = torch.cuda.Event(enable_timing=True)  # after score, before topk
            e3 = torch.cuda.Event(enable_timing=True)  # after topk, before remap
            e4 = torch.cuda.Event(enable_timing=True)  # after remap

            t0 = time.perf_counter_ns()
            e0.record()

            # Phase py_setup — shape reads + as_strided views only.
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
            e_alloc_start.record()

            # Phase alloc — torch.empty((B, max_scored), fp32).
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

            per_iter["py_setup_us"].append(e0.elapsed_time(e_alloc_start) * 1000.0)
            per_iter["alloc_us"].append(e_alloc_start.elapsed_time(e1) * 1000.0)
            per_iter["score_us"].append(e1.elapsed_time(e2) * 1000.0)
            per_iter["topk_us"].append(e2.elapsed_time(e3) * 1000.0)
            per_iter["remap_us"].append(e3.elapsed_time(e4) * 1000.0)
            per_iter["total_us"].append(e0.elapsed_time(e4) * 1000.0)
            per_iter["cpu_total_us"].append((t1 - t0) / 1000.0)

        return per_iter

    def timed_no_alloc(inputs, n_iters):
        """Ceiling after hoisting scores alloc + as_strided views out of the
        per-call path. Shows savings from caching view objects + buffer."""
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

        for _ in range(10):
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
        """Ceiling after replacing torch.topk with a cheap dummy indexer.

        Same everything else; torch.topk is replaced by a pre-allocated
        int64 zeros tensor. remap_kernel runs against it. Difference vs
        timed_no_alloc attributes a cost to torch.topk.
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
        topk_idx_stub = torch.zeros((batch_size, effective_topk), device=device, dtype=torch.int64)

        for _ in range(10):
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

    def pct(xs, q):
        if not xs:
            return 0.0
        xs = sorted(xs)
        k = int(round((q / 100) * (len(xs) - 1)))
        return xs[k]

    results = []
    for i, (label, wl) in enumerate(selected):
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

        # Targets get more iters for lower-noise p50/p90.
        n_iters = iters if label in {"small", "medium", "large"} else max(iters // 2, 30)

        try:
            per_iter = timed(inputs, n_iters=n_iters)
        except Exception as e:
            logger.exception(f"Workload {wl.uuid[:8]} timed failed: {e}")
            continue

        try:
            no_alloc_us = timed_no_alloc(inputs, n_iters=min(n_iters, 50))
        except Exception as e:
            logger.warning(f"Workload {wl.uuid[:8]} no_alloc failed: {e}")
            no_alloc_us = []

        try:
            stub_topk_us = stub_topk(inputs, n_iters=min(n_iters, 50))
        except Exception as e:
            logger.warning(f"Workload {wl.uuid[:8]} stub_topk failed: {e}")
            stub_topk_us = []

        result = {
            "uuid": wl.uuid,
            "label": label,
            "B": B,
            "max_num_pages": max_num_pages,
            "num_programs": num_prog,
            "num_pages_in_cache": P,
            "max_seq_len": max_sl,
            "sum_seq_len": sum_sl,
            "iters": len(per_iter["total_us"]),
        }
        for phase in ("py_setup_us", "alloc_us", "score_us", "topk_us",
                      "remap_us", "total_us", "cpu_total_us"):
            xs = per_iter[phase]
            result[f"{phase}_min"] = min(xs)
            result[f"{phase}_p50"] = pct(xs, 50)
            result[f"{phase}_p90"] = pct(xs, 90)
            result[f"{phase}_mean"] = statistics.mean(xs)
        result["no_alloc_p50_us"] = pct(no_alloc_us, 50) if no_alloc_us else 0.0
        result["no_alloc_p90_us"] = pct(no_alloc_us, 90) if no_alloc_us else 0.0
        result["stub_topk_p50_us"] = pct(stub_topk_us, 50) if stub_topk_us else 0.0
        result["stub_topk_p90_us"] = pct(stub_topk_us, 90) if stub_topk_us else 0.0

        results.append(result)
        logger.info(
            f"[{i+1}/{len(selected)}][{label}] {wl.uuid[:8]} B={B} pg={max_num_pages} "
            f"prog={num_prog} sum_sl={sum_sl} | "
            f"tot={result['total_us_p50']:5.1f} "
            f"py={result['py_setup_us_p50']:4.1f} "
            f"alloc={result['alloc_us_p50']:4.1f} "
            f"score={result['score_us_p50']:5.1f} "
            f"topk={result['topk_us_p50']:5.1f} "
            f"remap={result['remap_us_p50']:4.1f} "
            f"| cpu={result['cpu_total_us_p50']:5.1f} "
            f"no_alloc={result['no_alloc_p50_us']:5.1f} "
            f"stub={result['stub_topk_p50_us']:5.1f}"
        )

    return {"results": results}


@app.local_entrypoint()
def main(iters: int = 100, extra_stride: int = 16):
    print(f"Profiling iters={iters} extra_stride={extra_stride}")
    out = run_profile.remote(iters=iters, extra_stride=extra_stride)
    results = out["results"]
    if not results:
        print("No results.")
        return

    # Partition: targets vs context.
    targets = [r for r in results if r["label"] in {"small", "medium", "large"}]
    context = [r for r in results if r["label"] == "context"]

    print(f"\n== Target workloads (hand-picked) ==")
    print(f"{'label':<7} {'uuid':10} {'B':>3} {'pg':>4} {'prog':>5} {'sum_sl':>7} "
          f"{'tot':>7} {'py':>6} {'alloc':>6} {'score':>7} {'topk':>7} {'remap':>6} "
          f"{'cpu':>7} {'gap':>6}")
    for r in sorted(targets, key=lambda r: ["small", "medium", "large"].index(r["label"])):
        gap = r["cpu_total_us_p50"] - r["total_us_p50"]
        print(f"{r['label']:<7} {r['uuid'][:8]:10} {r['B']:>3} {r['max_num_pages']:>4} "
              f"{r['num_programs']:>5} {r['sum_seq_len']:>7} "
              f"{r['total_us_p50']:>7.1f} "
              f"{r['py_setup_us_p50']:>6.1f} "
              f"{r['alloc_us_p50']:>6.1f} "
              f"{r['score_us_p50']:>7.1f} "
              f"{r['topk_us_p50']:>7.1f} "
              f"{r['remap_us_p50']:>6.1f} "
              f"{r['cpu_total_us_p50']:>7.1f} "
              f"{gap:>+6.1f}")

    print(f"\n{'label':<7} {'uuid':10} {'no_alloc':>9} {'stub_topk':>10} "
          f"{'topk_cost':>10} {'non_topk':>9}")
    for r in sorted(targets, key=lambda r: ["small", "medium", "large"].index(r["label"])):
        topk_cost = r["no_alloc_p50_us"] - r["stub_topk_p50_us"]
        non_topk = r["stub_topk_p50_us"]
        print(f"{r['label']:<7} {r['uuid'][:8]:10} "
              f"{r['no_alloc_p50_us']:>9.1f} "
              f"{r['stub_topk_p50_us']:>10.1f} "
              f"{topk_cost:>10.1f} "
              f"{non_topk:>9.1f}")

    # Context agg
    totals = sorted(r["total_us_p50"] for r in context)
    if totals:
        median_total = totals[len(totals) // 2]
        small = [r for r in context if r["total_us_p50"] <= median_total]
        large = [r for r in context if r["total_us_p50"] > median_total]

        def agg(rs, key):
            xs = [r[key] for r in rs]
            if not xs:
                return (0, 0)
            xs_sorted = sorted(xs)
            p50 = xs_sorted[len(xs_sorted) // 2]
            p90 = xs_sorted[int(round(0.9 * (len(xs_sorted) - 1)))]
            return (p50, p90)

        print(f"\n== Context (stride={extra_stride}, n={len(context)}, "
              f"split at p50 tot={median_total:.1f} µs) ==")
        print(f"{'Phase':<15} {'small p50':>12} {'small p90':>12} "
              f"{'large p50':>12} {'large p90':>12}")
        for phase_key, label in [
            ("py_setup_us_p50", "py_setup"),
            ("alloc_us_p50", "alloc"),
            ("score_us_p50", "score_kernel"),
            ("topk_us_p50", "torch.topk"),
            ("remap_us_p50", "remap_kernel"),
            ("total_us_p50", "TOTAL (event)"),
            ("cpu_total_us_p50", "cpu_total"),
            ("no_alloc_p50_us", "no_alloc"),
            ("stub_topk_p50_us", "stub_topk"),
        ]:
            sp = agg(small, phase_key)
            lp = agg(large, phase_key)
            print(f"{label:<15} {sp[0]:>12.2f} {sp[1]:>12.2f} {lp[0]:>12.2f} {lp[1]:>12.2f}")

    import json
    out_path = PROJECT_ROOT / "experiments" / "profile_exp10_raw.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results → {out_path}")
