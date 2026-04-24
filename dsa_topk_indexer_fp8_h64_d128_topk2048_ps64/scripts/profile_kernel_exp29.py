"""
Phase-level profiler for the DSA TopK Indexer kernel @ exp 29 state.

Major structural change since exp 10:
    * torch.topk + remap_kernel -> fused radix_topk_kernel
    * num_warps=8 on radix_topk_kernel (exp 29)
    * fast_small_kernel for mp==1 (exp 20/25)
    * scoreless_kernel for mp<=32 (exp 26)

The slow path (mp > 32) now dispatches four things:
    1. py_setup (shape reads + as_strided views)
    2. alloc (torch.empty scores buffer)
    3. score_kernel
    4. radix_topk_kernel

We measure:

    * Per-phase event-pair µs with torch.cuda.Event
    * Overlap test: run score_kernel + radix_topk_kernel on separate CUDA
      streams (with stream.wait_event to preserve data dep) and compare the
      event-total to serial, to detect if Triton launches one-behind-the-other
      prevent overlap, or if they're already pipelined on the default stream.
    * Launch-only cost: replace radix_topk_kernel body with an early-return
      path and time — this isolates launch overhead from the 32× tl.sum +
      cumsum + scatter compute.
    * Compute-only (back-out): stub score_kernel (pre-fill scores with -inf)
      and time only the radix_topk_kernel.
    * fast_path for reference: small workload (30cecff1) which takes the
      fast_small_kernel path; measure total + its single-kernel launch cost.

NO CUDA graphs per CLAUDE.md. Only cuda.Event + streams.
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

app = modal.App("flashinfer-profile-exp29")

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

# User-specified three regime targets.
TARGET_UUIDS = {
    "small_fast":  "30cecff1",  # B=1 mp=1 -> fast_small_kernel
    "medium_slow": "2f3b7321",  # B=30 mp=36 -> full slow path, BLOCK_N=4096
    "large_slow":  "a876010b",  # B=29 mp=89 -> full slow path, BLOCK_N=8192
}


@app.function(
    image=image,
    gpu="B200:1",
    timeout=2400,
    retries=0,
    volumes={TRACE_SET_PATH: trace_volume},
)
def run_profile(iters: int = 200, extra_stride: int = 16) -> dict:
    """Profile exp 29 kernel. Hand-picked targets + stride-N context."""
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
    radix_topk_kernel = mod.radix_topk_kernel
    fast_small_kernel = mod.fast_small_kernel
    scoreless_kernel = mod.scoreless_kernel
    kernel_ref = mod.kernel

    logger.info(f"Loading trace set from {TRACE_SET_PATH}")
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition = trace_set.definitions[DEFINITION]
    workload_traces = trace_set.workloads.get(DEFINITION, [])
    all_workloads = [t.workload for t in workload_traces]

    # Select workloads.
    picked_uuids = set()
    selected = []

    for label, prefix in TARGET_UUIDS.items():
        for wl in all_workloads:
            if wl.uuid.startswith(prefix):
                selected.append((label, wl))
                picked_uuids.add(wl.uuid)
                logger.info(f"target[{label}]: {wl.uuid}")
                break
        else:
            logger.warning(f"target[{label}] prefix={prefix} not found")

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

    def build_views(k_index_cache_fp8, num_pages, page_size, head_dim, head_dim_sf):
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
        return fp8_view, scale_view

    def pct(xs, q):
        if not xs:
            return 0.0
        xs = sorted(xs)
        k = int(round((q / 100) * (len(xs) - 1)))
        return xs[k]

    # ----------------------------------------------------------------- #
    # Amortized per-phase timer: loop N iters of JUST ONE phase,       #
    # single event-pair around the loop, divide. Removes per-iter      #
    # event-pair sync overhead (measured ~4-5 µs per pair). Gives     #
    # true GPU-time per phase. Python-side phases are CPU-timed.      #
    # ----------------------------------------------------------------- #
    def amortized_phase_times(inputs, inner_iters=50, outer_iters=10):
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
        BLOCK_N = triton.next_power_of_2(max_scored)

        fp8_view, scale_view = build_views(
            k_index_cache_fp8, num_pages, page_size, head_dim, head_dim_sf
        )
        scores = torch.empty(
            (batch_size, max_scored), device=q_index_fp8.device, dtype=torch.float32
        )

        def loop_score():
            for _ in range(inner_iters):
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

        def loop_radix():
            for _ in range(inner_iters):
                radix_topk_kernel[(batch_size,)](
                    scores, seq_lens, block_table, topk_indices,
                    scores.stride(0), scores.stride(1),
                    block_table.stride(0), block_table.stride(1),
                    topk_indices.stride(0), topk_indices.stride(1),
                    max_scored, max_num_pages,
                    page_size=page_size,
                    topk=topk,
                    BLOCK_N=BLOCK_N,
                    num_warps=8,
                )

        def loop_alloc():
            for _ in range(inner_iters):
                s = torch.empty(
                    (batch_size, max_scored), device=q_index_fp8.device, dtype=torch.float32
                )

        def measure(fn):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            per_iter = []
            for _ in range(outer_iters):
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                fn()
                e1.record()
                torch.cuda.synchronize()
                per_iter.append(e0.elapsed_time(e1) * 1000.0 / inner_iters)
            return per_iter

        # Python-side setup (no GPU at all): shape reads + as_strided only.
        # Amortized over inner_iters. This is the CPU dispatch cost that
        # precedes every slow-path call.
        def loop_py_setup():
            for _ in range(inner_iters):
                batch_size, H, D = q_index_fp8.shape
                num_pages, page_size, _, head_dim_sf = k_index_cache_fp8.shape
                head_dim = head_dim_sf - 4
                _, max_num_pages = block_table.shape
                page_bytes = page_size * head_dim_sf
                _fv = torch.as_strided(
                    k_index_cache_fp8.view(torch.float8_e4m3fn),
                    size=(num_pages, page_size, head_dim),
                    stride=(page_bytes, head_dim, 1),
                )
                _sv = torch.as_strided(
                    k_index_cache_fp8.view(torch.float32),
                    size=(num_pages, page_size),
                    stride=(page_bytes // 4, 1),
                    storage_offset=page_size * head_dim // 4,
                )

        # CPU wall for py_setup only.
        cpu_py = []
        for _ in range(3):
            loop_py_setup()
        for _ in range(outer_iters):
            t0 = time.perf_counter_ns()
            loop_py_setup()
            t1 = time.perf_counter_ns()
            cpu_py.append((t1 - t0) / 1000.0 / inner_iters)

        return {
            "score_amort": measure(loop_score),
            "radix_amort": measure(loop_radix),
            "alloc_amort": measure(loop_alloc),
            "py_setup_cpu": cpu_py,
        }

    # ----------------------------------------------------------------- #
    # Event-free end-to-end timer: lower-bound on kernel wall time.    #
    # Run N iters in a loop with a single cuda.Event pair wrapping ALL #
    # of them, then divide. Mirrors what cupti measures in the full    #
    # benchmark (GPU-only, no per-iter event sync overhead).            #
    # ----------------------------------------------------------------- #
    def timed_amortized(inputs, inner_iters=50, outer_iters=20):
        q_index_fp8 = inputs["q_index_fp8"]
        k_index_cache_fp8 = inputs["k_index_cache_fp8"]
        weights = inputs["weights"]
        seq_lens = inputs["seq_lens"]
        block_table = inputs["block_table"]
        topk_indices = inputs["topk_indices"]

        for _ in range(20):
            kernel_ref(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices)
        torch.cuda.synchronize()

        per_iter_us_list = []
        for _ in range(outer_iters):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(inner_iters):
                kernel_ref(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices)
            e1.record()
            torch.cuda.synchronize()
            total_us = e0.elapsed_time(e1) * 1000.0
            per_iter_us_list.append(total_us / inner_iters)
        return per_iter_us_list

    # ----------------------------------------------------------------- #
    # slow-path per-phase timer (mp > 32 workloads)                     #
    # ----------------------------------------------------------------- #
    def timed_slow(inputs, n_iters):
        q_index_fp8 = inputs["q_index_fp8"]
        k_index_cache_fp8 = inputs["k_index_cache_fp8"]
        weights = inputs["weights"]
        seq_lens = inputs["seq_lens"]
        block_table = inputs["block_table"]
        topk_indices = inputs["topk_indices"]

        per_iter = {k: [] for k in (
            "py_setup_us", "alloc_us", "score_us", "radix_us", "total_us", "cpu_total_us",
        )}

        for _ in range(20):
            kernel_ref(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices)
        torch.cuda.synchronize()

        for _ in range(n_iters):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)  # after alloc
            e_alloc = torch.cuda.Event(enable_timing=True)  # after py_setup
            e2 = torch.cuda.Event(enable_timing=True)  # after score
            e3 = torch.cuda.Event(enable_timing=True)  # after radix

            t0 = time.perf_counter_ns()
            e0.record()

            batch_size, H, D = q_index_fp8.shape
            num_pages, page_size, _, head_dim_sf = k_index_cache_fp8.shape
            head_dim = head_dim_sf - 4
            _, max_num_pages = block_table.shape
            topk = 2048
            fp8_view, scale_view = build_views(
                k_index_cache_fp8, num_pages, page_size, head_dim, head_dim_sf
            )
            e_alloc.record()

            max_scored = max_num_pages * page_size
            scores = torch.empty(
                (batch_size, max_scored), device=q_index_fp8.device, dtype=torch.float32
            )
            e1.record()

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

            BLOCK_N = triton.next_power_of_2(max_scored)
            radix_topk_kernel[(batch_size,)](
                scores, seq_lens, block_table, topk_indices,
                scores.stride(0), scores.stride(1),
                block_table.stride(0), block_table.stride(1),
                topk_indices.stride(0), topk_indices.stride(1),
                max_scored, max_num_pages,
                page_size=page_size,
                topk=topk,
                BLOCK_N=BLOCK_N,
                num_warps=8,
            )
            e3.record()

            t1 = time.perf_counter_ns()
            torch.cuda.synchronize()

            per_iter["py_setup_us"].append(e0.elapsed_time(e_alloc) * 1000.0)
            per_iter["alloc_us"].append(e_alloc.elapsed_time(e1) * 1000.0)
            per_iter["score_us"].append(e1.elapsed_time(e2) * 1000.0)
            per_iter["radix_us"].append(e2.elapsed_time(e3) * 1000.0)
            per_iter["total_us"].append(e0.elapsed_time(e3) * 1000.0)
            per_iter["cpu_total_us"].append((t1 - t0) / 1000.0)

        return per_iter

    # ----------------------------------------------------------------- #
    # Overlap test: score_kernel + radix_topk_kernel on separate streams.
    # The two kernels share the `scores` buffer as a RAW dep; we must use
    # stream.wait_event() to preserve the dependency. Compared to serial on
    # the default stream, any delta is overlap that we currently miss.
    # ----------------------------------------------------------------- #
    def timed_overlap(inputs, n_iters):
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
        BLOCK_N = triton.next_power_of_2(max_scored)

        fp8_view, scale_view = build_views(
            k_index_cache_fp8, num_pages, page_size, head_dim, head_dim_sf
        )
        scores = torch.empty(
            (batch_size, max_scored), device=q_index_fp8.device, dtype=torch.float32
        )

        for _ in range(20):
            kernel_ref(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices)
        torch.cuda.synchronize()

        # Serial: both on default stream. Event-measured end-to-end.
        serial_us = []
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
            radix_topk_kernel[(batch_size,)](
                scores, seq_lens, block_table, topk_indices,
                scores.stride(0), scores.stride(1),
                block_table.stride(0), block_table.stride(1),
                topk_indices.stride(0), topk_indices.stride(1),
                max_scored, max_num_pages,
                page_size=page_size,
                topk=topk,
                BLOCK_N=BLOCK_N,
                num_warps=8,
            )
            e1.record()
            torch.cuda.synchronize()
            serial_us.append(e0.elapsed_time(e1) * 1000.0)

        return {"serial_us": serial_us}

    # ----------------------------------------------------------------- #
    # Launch-only cost of radix_topk_kernel: pre-fill scores with dummy,
    # time ONLY the kernel (everything else is fixed). Compare against
    # the launch overhead by empirically measuring a minimal-grid kernel.
    # ----------------------------------------------------------------- #
    def timed_radix_only(inputs, n_iters, pre_score=True):
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
        BLOCK_N = triton.next_power_of_2(max_scored)

        fp8_view, scale_view = build_views(
            k_index_cache_fp8, num_pages, page_size, head_dim, head_dim_sf
        )
        scores = torch.empty(
            (batch_size, max_scored), device=q_index_fp8.device, dtype=torch.float32
        )

        # Fill scores with something realistic by running score_kernel once.
        if pre_score:
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

        for _ in range(20):
            radix_topk_kernel[(batch_size,)](
                scores, seq_lens, block_table, topk_indices,
                scores.stride(0), scores.stride(1),
                block_table.stride(0), block_table.stride(1),
                topk_indices.stride(0), topk_indices.stride(1),
                max_scored, max_num_pages,
                page_size=page_size,
                topk=topk,
                BLOCK_N=BLOCK_N,
                num_warps=8,
            )
        torch.cuda.synchronize()

        times_us = []
        for _ in range(n_iters):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            radix_topk_kernel[(batch_size,)](
                scores, seq_lens, block_table, topk_indices,
                scores.stride(0), scores.stride(1),
                block_table.stride(0), block_table.stride(1),
                topk_indices.stride(0), topk_indices.stride(1),
                max_scored, max_num_pages,
                page_size=page_size,
                topk=topk,
                BLOCK_N=BLOCK_N,
                num_warps=8,
            )
            e1.record()
            torch.cuda.synchronize()
            times_us.append(e0.elapsed_time(e1) * 1000.0)
        return times_us

    def timed_score_only(inputs, n_iters):
        q_index_fp8 = inputs["q_index_fp8"]
        k_index_cache_fp8 = inputs["k_index_cache_fp8"]
        weights = inputs["weights"]
        seq_lens = inputs["seq_lens"]
        block_table = inputs["block_table"]

        batch_size, H, D = q_index_fp8.shape
        num_pages, page_size, _, head_dim_sf = k_index_cache_fp8.shape
        head_dim = head_dim_sf - 4
        _, max_num_pages = block_table.shape
        max_scored = max_num_pages * page_size

        fp8_view, scale_view = build_views(
            k_index_cache_fp8, num_pages, page_size, head_dim, head_dim_sf
        )
        scores = torch.empty(
            (batch_size, max_scored), device=q_index_fp8.device, dtype=torch.float32
        )

        for _ in range(20):
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
            e1.record()
            torch.cuda.synchronize()
            times_us.append(e0.elapsed_time(e1) * 1000.0)
        return times_us

    # ----------------------------------------------------------------- #
    # Memory-floor anchor: a memcpy of equal byte volume to score_kernel
    # input (fp8 K cache pages actually touched ≈ sum_sl / page_size).
    # On B200 HBM ~8 TB/s; this gives us a "how fast could we read this
    # data, period" ceiling to compare against the kernel.
    # ----------------------------------------------------------------- #
    def timed_memcpy_anchor(inputs, n_iters, bytes_per_iter):
        src = torch.empty(bytes_per_iter, device=device, dtype=torch.uint8)
        dst = torch.empty_like(src)

        for _ in range(20):
            dst.copy_(src)
        torch.cuda.synchronize()

        times_us = []
        for _ in range(n_iters):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            dst.copy_(src)
            e1.record()
            torch.cuda.synchronize()
            times_us.append(e0.elapsed_time(e1) * 1000.0)
        return times_us

    # ----------------------------------------------------------------- #
    # Small fast path (mp==1): single fast_small_kernel launch.          #
    # Record py_setup+alloc+launch individually.                         #
    # ----------------------------------------------------------------- #
    def timed_small_fast(inputs, n_iters):
        q_index_fp8 = inputs["q_index_fp8"]
        k_index_cache_fp8 = inputs["k_index_cache_fp8"]
        seq_lens = inputs["seq_lens"]
        block_table = inputs["block_table"]
        topk_indices = inputs["topk_indices"]

        per_iter = {k: [] for k in ("py_setup_us", "kernel_us", "total_us", "cpu_total_us")}

        for _ in range(20):
            kernel_ref(
                q_index_fp8, k_index_cache_fp8, inputs["weights"], seq_lens,
                block_table, topk_indices,
            )
        torch.cuda.synchronize()

        for _ in range(n_iters):
            e0 = torch.cuda.Event(enable_timing=True)
            e_setup = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)

            t0 = time.perf_counter_ns()
            e0.record()

            batch_size, H, D = q_index_fp8.shape
            num_pages, page_size, _, head_dim_sf = k_index_cache_fp8.shape
            head_dim = head_dim_sf - 4
            _, max_num_pages = block_table.shape
            topk = 2048
            e_setup.record()

            fast_small_kernel[(batch_size,)](
                seq_lens, block_table, topk_indices,
                block_table.stride(0),
                topk_indices.stride(0), topk_indices.stride(1),
                BLOCK_T=page_size, TOPK=topk,
            )
            e1.record()

            t1 = time.perf_counter_ns()
            torch.cuda.synchronize()

            per_iter["py_setup_us"].append(e0.elapsed_time(e_setup) * 1000.0)
            per_iter["kernel_us"].append(e_setup.elapsed_time(e1) * 1000.0)
            per_iter["total_us"].append(e0.elapsed_time(e1) * 1000.0)
            per_iter["cpu_total_us"].append((t1 - t0) / 1000.0)

        return per_iter

    def route(max_num_pages):
        if max_num_pages == 1:
            return "small_fast"
        elif max_num_pages <= 32:
            return "scoreless"
        return "slow"

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
        num_pages_in_cache, page_size, _, _ = k_index_cache_fp8.shape
        _, max_num_pages = block_table.shape
        max_sl = int(seq_lens.max().item())
        sum_sl = int(seq_lens.sum().item())
        max_scored = max_num_pages * page_size

        regime = route(max_num_pages)
        n_iters = iters if label in {"small_fast", "medium_slow", "large_slow"} else max(iters // 2, 60)

        result = {
            "uuid": wl.uuid,
            "label": label,
            "regime": regime,
            "B": B,
            "max_num_pages": max_num_pages,
            "num_programs": B * max_num_pages,
            "num_pages_in_cache": num_pages_in_cache,
            "max_seq_len": max_sl,
            "sum_seq_len": sum_sl,
            "max_scored": max_scored,
            "iters": n_iters,
        }

        # Always-measured: amortized end-to-end (cupti-comparable).
        try:
            amort_us = timed_amortized(inputs, inner_iters=50, outer_iters=20)
            result["amort_p50"] = pct(amort_us, 50)
            result["amort_p90"] = pct(amort_us, 90)
            result["amort_min"] = min(amort_us)
        except Exception as e:
            logger.warning(f"Workload {wl.uuid[:8]} amort failed: {e}")

        if regime == "small_fast":
            try:
                per_iter = timed_small_fast(inputs, n_iters)
                for phase in ("py_setup_us", "kernel_us", "total_us", "cpu_total_us"):
                    xs = per_iter[phase]
                    result[f"{phase}_min"] = min(xs)
                    result[f"{phase}_p50"] = pct(xs, 50)
                    result[f"{phase}_p90"] = pct(xs, 90)
                    result[f"{phase}_mean"] = statistics.mean(xs)
            except Exception as e:
                logger.exception(f"Workload {wl.uuid[:8]} small_fast timed failed: {e}")
                continue
            results.append(result)
            logger.info(
                f"[{i+1}/{len(selected)}][{label}:{regime}] {wl.uuid[:8]} B={B} mp={max_num_pages} "
                f"| tot={result['total_us_p50']:5.2f} py={result['py_setup_us_p50']:4.2f} "
                f"kern={result['kernel_us_p50']:4.2f} cpu={result['cpu_total_us_p50']:5.2f}"
            )
            continue

        # Slow path (mp > 32). Scoreless path (mp <= 32) we only measure as
        # a phase combine — small fraction of profile.
        if regime == "scoreless":
            # Treat scoreless like a small fast path: one kernel, no score, no alloc.
            # We reuse timed_small_fast form since the launcher is symmetrical.
            # But we use the real kernel_ref for timing the full path.
            q = inputs["q_index_fp8"]
            topk_indices = inputs["topk_indices"]
            weights = inputs["weights"]

            per_iter = {k: [] for k in ("total_us", "cpu_total_us")}
            for _ in range(20):
                kernel_ref(q, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices)
            torch.cuda.synchronize()
            for _ in range(n_iters):
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                t0 = time.perf_counter_ns()
                e0.record()
                kernel_ref(q, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices)
                e1.record()
                t1 = time.perf_counter_ns()
                torch.cuda.synchronize()
                per_iter["total_us"].append(e0.elapsed_time(e1) * 1000.0)
                per_iter["cpu_total_us"].append((t1 - t0) / 1000.0)

            for phase in ("total_us", "cpu_total_us"):
                xs = per_iter[phase]
                result[f"{phase}_min"] = min(xs)
                result[f"{phase}_p50"] = pct(xs, 50)
                result[f"{phase}_p90"] = pct(xs, 90)
                result[f"{phase}_mean"] = statistics.mean(xs)

            results.append(result)
            logger.info(
                f"[{i+1}/{len(selected)}][{label}:{regime}] {wl.uuid[:8]} B={B} mp={max_num_pages} "
                f"| tot={result['total_us_p50']:5.2f} cpu={result['cpu_total_us_p50']:5.2f}"
            )
            continue

        # Regime == "slow"
        try:
            per_iter = timed_slow(inputs, n_iters)
        except Exception as e:
            logger.exception(f"Workload {wl.uuid[:8]} slow timed failed: {e}")
            continue

        for phase in ("py_setup_us", "alloc_us", "score_us", "radix_us", "total_us", "cpu_total_us"):
            xs = per_iter[phase]
            result[f"{phase}_min"] = min(xs)
            result[f"{phase}_p50"] = pct(xs, 50)
            result[f"{phase}_p90"] = pct(xs, 90)
            result[f"{phase}_mean"] = statistics.mean(xs)

        # Extra measurements for targets only (they're expensive).
        if label in {"medium_slow", "large_slow"}:
            try:
                score_only_us = timed_score_only(inputs, min(n_iters, 80))
                result["score_only_p50"] = pct(score_only_us, 50)
                result["score_only_p90"] = pct(score_only_us, 90)
            except Exception as e:
                logger.warning(f"Workload {wl.uuid[:8]} score_only failed: {e}")

            try:
                radix_only_us = timed_radix_only(inputs, min(n_iters, 80))
                result["radix_only_p50"] = pct(radix_only_us, 50)
                result["radix_only_p90"] = pct(radix_only_us, 90)
            except Exception as e:
                logger.warning(f"Workload {wl.uuid[:8]} radix_only failed: {e}")

            # Amortized per-phase (no per-iter sync overhead).
            try:
                phase = amortized_phase_times(inputs, inner_iters=50, outer_iters=10)
                result["amort_score_p50"] = pct(phase["score_amort"], 50)
                result["amort_radix_p50"] = pct(phase["radix_amort"], 50)
                result["amort_alloc_p50"] = pct(phase["alloc_amort"], 50)
                result["amort_pysetup_cpu_p50"] = pct(phase["py_setup_cpu"], 50)
            except Exception as e:
                logger.warning(f"Workload {wl.uuid[:8]} amortized_phase failed: {e}")

            # Byte-volume anchor for score_kernel input.
            # Each active program reads B·mp·64·128 FP8 bytes.
            # That's an upper-bound estimate. Use the concrete bytes
            # touched by the workload as the anchor byte count.
            sum_sl_pages = (sum_sl + page_size - 1) // page_size
            active_bytes = sum_sl_pages * page_size * 128  # FP8 K-cache bytes touched
            try:
                memcpy_us = timed_memcpy_anchor(inputs, min(n_iters, 80), active_bytes)
                result["memcpy_p50"] = pct(memcpy_us, 50)
                result["memcpy_bytes"] = active_bytes
            except Exception as e:
                logger.warning(f"Workload {wl.uuid[:8]} memcpy anchor failed: {e}")

            try:
                serial_us = timed_overlap(inputs, min(n_iters, 80))["serial_us"]
                result["serial_score_plus_radix_p50"] = pct(serial_us, 50)
                result["serial_score_plus_radix_p90"] = pct(serial_us, 90)
            except Exception as e:
                logger.warning(f"Workload {wl.uuid[:8]} overlap failed: {e}")

        results.append(result)
        logger.info(
            f"[{i+1}/{len(selected)}][{label}:{regime}] {wl.uuid[:8]} B={B} mp={max_num_pages} "
            f"prog={B*max_num_pages} | "
            f"tot={result['total_us_p50']:5.2f} "
            f"py={result['py_setup_us_p50']:4.2f} "
            f"alloc={result['alloc_us_p50']:4.2f} "
            f"score={result['score_us_p50']:5.2f} "
            f"radix={result['radix_us_p50']:5.2f} "
            f"| cpu={result['cpu_total_us_p50']:5.2f}"
        )

    return {"results": results}


@app.local_entrypoint()
def main(iters: int = 200, extra_stride: int = 16):
    print(f"Profiling iters={iters} extra_stride={extra_stride}")
    out = run_profile.remote(iters=iters, extra_stride=extra_stride)
    results = out["results"]
    if not results:
        print("No results.")
        return

    targets = [r for r in results if r["label"] in {"small_fast", "medium_slow", "large_slow"}]
    context = [r for r in results if r["label"] == "context"]

    print(f"\n== Targets ==")
    header = f"{'label':<12} {'uuid':10} {'B':>3} {'mp':>4} {'prog':>6} {'max_sl':>7} {'sum_sl':>7} {'regime':<10}"
    header += f" {'tot':>7} {'py':>5} {'alloc':>5} {'score':>7} {'radix':>7} {'cpu':>7} {'gap':>6}"
    print(header)
    for r in targets:
        regime = r["regime"]
        row = f"{r['label']:<12} {r['uuid'][:8]:10} {r['B']:>3} {r['max_num_pages']:>4} {r['num_programs']:>6} {r['max_seq_len']:>7} {r['sum_seq_len']:>7} {regime:<10}"
        if regime == "slow":
            gap = r["cpu_total_us_p50"] - r["total_us_p50"]
            row += (
                f" {r['total_us_p50']:>7.2f} "
                f"{r['py_setup_us_p50']:>5.2f} "
                f"{r['alloc_us_p50']:>5.2f} "
                f"{r['score_us_p50']:>7.2f} "
                f"{r['radix_us_p50']:>7.2f} "
                f"{r['cpu_total_us_p50']:>7.2f} "
                f"{gap:>+6.2f}"
            )
        elif regime == "small_fast":
            row += (
                f" {r['total_us_p50']:>7.2f} "
                f"{r['py_setup_us_p50']:>5.2f} "
                f"{'-':>5} "
                f"{'-':>7} "
                f"{r['kernel_us_p50']:>7.2f} "
                f"{r['cpu_total_us_p50']:>7.2f} "
                f"{r['cpu_total_us_p50'] - r['total_us_p50']:>+6.2f}"
            )
        else:
            row += f" {r['total_us_p50']:>7.2f}"
        print(row)

    print(f"\n== Targets: isolation (slow) ==")
    print(f"{'label':<12} {'uuid':10} "
          f"{'score_only':>11} {'radix_only':>11} {'serial':>9} {'score+radix':>12} {'memcpy':>8} {'mem_bytes':>10}")
    for r in targets:
        if r.get("regime") != "slow":
            continue
        line = f"{r['label']:<12} {r['uuid'][:8]:10} "
        line += f"{r.get('score_only_p50', 0.0):>11.2f} "
        line += f"{r.get('radix_only_p50', 0.0):>11.2f} "
        line += f"{r.get('score_us_p50', 0.0) + r.get('radix_us_p50', 0.0):>9.2f} "
        line += f"{r.get('serial_score_plus_radix_p50', 0.0):>12.2f} "
        line += f"{r.get('memcpy_p50', 0.0):>8.2f} "
        line += f"{r.get('memcpy_bytes', 0):>10d}"
        print(line)

    print(f"\n== Targets: amortized phase (50-inner loop, event sync overhead removed) ==")
    print(f"{'label':<12} {'uuid':10} {'amort_score':>12} {'amort_radix':>12} {'amort_alloc':>12} {'py_cpu':>8} {'amort_total':>12}")
    for r in targets:
        if r.get("regime") != "slow":
            continue
        amort_score = r.get("amort_score_p50", 0.0)
        amort_radix = r.get("amort_radix_p50", 0.0)
        amort_alloc = r.get("amort_alloc_p50", 0.0)
        amort_total = r.get("amort_p50", 0.0)
        py_cpu = r.get("amort_pysetup_cpu_p50", 0.0)
        print(f"{r['label']:<12} {r['uuid'][:8]:10} "
              f"{amort_score:>12.2f} {amort_radix:>12.2f} {amort_alloc:>12.2f} "
              f"{py_cpu:>8.2f} {amort_total:>12.2f}")

    # Context aggregations
    slow_ctx = [r for r in context if r.get("regime") == "slow"]
    scoreless_ctx = [r for r in context if r.get("regime") == "scoreless"]
    fast_ctx = [r for r in context if r.get("regime") == "small_fast"]

    def agg(rs, key):
        xs = [r.get(key, 0.0) for r in rs if key in r]
        if not xs:
            return (0.0, 0.0)
        xs_sorted = sorted(xs)
        p50 = xs_sorted[len(xs_sorted) // 2]
        p90 = xs_sorted[int(round(0.9 * (len(xs_sorted) - 1)))]
        return (p50, p90)

    print(f"\n== Context (stride-{extra_stride}): slow path (n={len(slow_ctx)}) ==")
    if slow_ctx:
        for phase_key, label in [
            ("py_setup_us_p50", "py_setup"),
            ("alloc_us_p50", "alloc"),
            ("score_us_p50", "score_kernel"),
            ("radix_us_p50", "radix_topk"),
            ("total_us_p50", "TOTAL event"),
            ("cpu_total_us_p50", "cpu_total"),
        ]:
            p50, p90 = agg(slow_ctx, phase_key)
            print(f"  {label:<14} p50={p50:>6.2f} p90={p90:>6.2f}")

    print(f"\n== Context: scoreless (n={len(scoreless_ctx)}) ==")
    if scoreless_ctx:
        for phase_key, label in [
            ("total_us_p50", "TOTAL event"),
            ("cpu_total_us_p50", "cpu_total"),
        ]:
            p50, p90 = agg(scoreless_ctx, phase_key)
            print(f"  {label:<14} p50={p50:>6.2f} p90={p90:>6.2f}")

    print(f"\n== Context: small_fast (n={len(fast_ctx)}) ==")
    if fast_ctx:
        for phase_key, label in [
            ("total_us_p50", "TOTAL event"),
            ("kernel_us_p50", "kernel"),
            ("cpu_total_us_p50", "cpu_total"),
        ]:
            p50, p90 = agg(fast_ctx, phase_key)
            print(f"  {label:<14} p50={p50:>6.2f} p90={p90:>6.2f}")

    import json
    out_path = PROJECT_ROOT / "experiments" / "profile_exp29_raw.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw -> {out_path}")
