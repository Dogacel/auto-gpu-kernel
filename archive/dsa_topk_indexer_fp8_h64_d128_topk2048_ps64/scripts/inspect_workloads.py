"""
Workload inspection for DSA TopK Indexer.

Loads ALL 128 workloads of the `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`
definition and dumps statistics about input shapes/values. The goal is to
find exploitable structure (batch_size skew, seq_lens skew, utilization,
block_table contiguity, effective_topk < topk, max_num_pages distribution).

This script does NOT run the kernel. Pure data shape characterization.

Output is written to experiments/workload_profile.md by the local entrypoint.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# DNS patch for sandboxed networks (same pattern as profile_kernel.py).
try:
    from scripts._dns_patch import patch as _patch_dns
    _patch_dns()
except Exception:
    pass

import modal

app = modal.App("flashinfer-inspect-workloads")

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
)

DEFINITION = "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64"


@app.function(
    image=image,
    gpu="B200:1",
    timeout=1800,
    retries=0,
    volumes={TRACE_SET_PATH: trace_volume},
)
def inspect_all() -> dict:
    """Load all workloads, compute per-workload statistics, return summary dict."""
    import logging
    import torch
    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition = trace_set.definitions[DEFINITION]
    workload_traces = trace_set.workloads.get(DEFINITION, [])
    workloads = [t.workload for t in workload_traces]
    logger.info(f"Total workloads: {len(workloads)}")

    device = torch.device("cuda")
    page_size = 64
    topk = 2048

    results = []

    for i, wl in enumerate(workloads):
        safe = load_safetensors(definition, wl, trace_set_root=trace_set.root)
        vals = gen_inputs(definition, wl, device=str(device), safe_tensors=safe)
        names = list(definition.inputs.keys())
        inputs = dict(zip(names, vals))

        q = inputs["q_index_fp8"]             # [B, 64, 128]
        kv = inputs["k_index_cache_fp8"]      # [P, 64, 1, 132]
        weights = inputs["weights"]            # [B, 64]
        seq_lens = inputs["seq_lens"]          # [B]
        block_table = inputs["block_table"]    # [B, max_num_pages]

        B = q.shape[0]
        num_pages_in_cache = kv.shape[0]
        max_num_pages = block_table.shape[1]

        sl_cpu = seq_lens.to(torch.int64).cpu()
        sl_list = sl_cpu.tolist()
        max_sl = int(sl_cpu.max().item()) if B > 0 else 0
        min_sl = int(sl_cpu.min().item()) if B > 0 else 0
        sum_sl = int(sl_cpu.sum().item())
        # Per-batch num_pages_for_seq = ceil(seq_len / page_size)
        num_pages_per_b = [(s + page_size - 1) // page_size for s in sl_list]
        max_pg_per_b = max(num_pages_per_b) if num_pages_per_b else 0
        sum_pg_used = sum(num_pages_per_b)

        # Grid size (score_kernel outer loop axis).
        num_programs = B * max_num_pages
        # Effective active programs.
        active_programs = sum(num_pages_per_b)
        # Block-uniform early-return fraction (% of programs that return empty).
        early_return_frac = 1.0 - (active_programs / num_programs) if num_programs > 0 else 0.0

        # Utilization: sum_sl / (B * max_num_pages * page_size). How many token
        # slots in the grid are real tokens vs. padding.
        slot_capacity = B * max_num_pages * page_size
        util = sum_sl / slot_capacity if slot_capacity > 0 else 0.0

        # Intra-batch seq_len skew.
        skew_max_over_min = max_sl / max(min_sl, 1) if min_sl > 0 else float("inf")
        skew_sum_over_capacity = sum_sl / (B * max_sl) if (B > 0 and max_sl > 0) else 0.0

        # effective_topk = min(topk, sum_sl). Per batch: actual_topk[b] = min(topk, sl[b]).
        # effective_topk across the batch (what torch.topk actually picks):
        # min(topk, max_scored=max_num_pages*page_size).
        effective_topk_kernel = min(topk, max_num_pages * page_size)
        # Per-batch effective topk (what meaningful k each row has):
        per_batch_actual_topk = [min(topk, s) for s in sl_list]
        max_actual_topk = max(per_batch_actual_topk) if per_batch_actual_topk else 0
        all_below_2048 = all(t < topk for t in per_batch_actual_topk)

        # block_table contiguity. For each row of block_table (only first
        # num_pages_per_b[b] entries are valid), measure:
        #   - fraction of consecutive pairs where page[i+1] == page[i]+1
        #   - also count total "runs" (distinct contiguous chunks)
        bt_cpu = block_table.to(torch.int64).cpu()
        total_pairs = 0
        contig_pairs = 0
        per_batch_contig_frac = []
        # Cross-item reuse: same page_id referenced by multiple batch rows.
        all_pages_used = []
        for b in range(B):
            npg = num_pages_per_b[b]
            if npg <= 1:
                per_batch_contig_frac.append(1.0)
                if npg == 1:
                    all_pages_used.append(int(bt_cpu[b, 0].item()))
                continue
            row = bt_cpu[b, :npg].tolist()
            all_pages_used.extend(row)
            pairs = npg - 1
            cpairs = sum(1 for j in range(pairs) if row[j + 1] == row[j] + 1)
            total_pairs += pairs
            contig_pairs += cpairs
            per_batch_contig_frac.append(cpairs / pairs if pairs > 0 else 1.0)
        contig_frac_overall = contig_pairs / total_pairs if total_pairs > 0 else 1.0
        unique_pages_used = len(set(all_pages_used))
        total_pages_used = len(all_pages_used)
        # Reuse factor = total / unique. >1 means some page is shared across batch items.
        reuse_factor = total_pages_used / max(unique_pages_used, 1)

        # block_table tail: values beyond num_pages_per_b[b]. Could be garbage
        # (positive), zero, or -1. Determines whether the early-return in the
        # score kernel is necessary or if we could rely on seq_lens masking only.
        tail_vals = []
        for b in range(B):
            npg = num_pages_per_b[b]
            if npg < max_num_pages:
                tail = bt_cpu[b, npg:]
                tail_vals.extend(tail.tolist())
        if tail_vals:
            tail_all_zero = all(v == 0 for v in tail_vals)
            tail_all_neg1 = all(v == -1 for v in tail_vals)
            tail_all_valid_page = all(0 <= v < num_pages_in_cache for v in tail_vals)
            tail_min = min(tail_vals)
            tail_max = max(tail_vals)
        else:
            tail_all_zero = True
            tail_all_neg1 = False
            tail_all_valid_page = True
            tail_min = 0
            tail_max = 0

        # Value sparsity — weights: how often are weights[b,h] == 0? (cheap skip path.)
        weights_zero_frac = float((weights == 0).float().mean().item())

        # seq_lens==0 and seq_lens<topk counts within a batch:
        num_sl_zero = sum(1 for s in sl_list if s == 0)
        num_sl_lt_topk = sum(1 for s in sl_list if s < topk)
        num_sl_lt_page = sum(1 for s in sl_list if s < page_size)

        result = {
            "uuid": wl.uuid,
            "B": B,
            "max_num_pages": max_num_pages,
            "num_programs": num_programs,
            "num_pages_in_cache": num_pages_in_cache,
            "active_programs": active_programs,
            "early_return_frac": early_return_frac,
            "seq_lens": sl_list,
            "min_sl": min_sl,
            "max_sl": max_sl,
            "sum_sl": sum_sl,
            "skew_max_over_min": skew_max_over_min,
            "skew_sum_over_capacity": skew_sum_over_capacity,
            "utilization": util,
            "effective_topk_kernel": effective_topk_kernel,
            "max_actual_topk": max_actual_topk,
            "all_below_2048": all_below_2048,
            "num_sl_zero": num_sl_zero,
            "num_sl_lt_topk": num_sl_lt_topk,
            "num_sl_lt_page": num_sl_lt_page,
            "per_batch_actual_topk": per_batch_actual_topk,
            "contig_frac_overall": contig_frac_overall,
            "per_batch_contig_frac": per_batch_contig_frac,
            "unique_pages_used": unique_pages_used,
            "total_pages_used": total_pages_used,
            "reuse_factor": reuse_factor,
            "tail_all_zero": tail_all_zero,
            "tail_all_neg1": tail_all_neg1,
            "tail_all_valid_page": tail_all_valid_page,
            "tail_min": tail_min,
            "tail_max": tail_max,
            "weights_zero_frac": weights_zero_frac,
            "num_pages_per_b": num_pages_per_b,
        }
        results.append(result)

        if (i + 1) % 16 == 0 or i == len(workloads) - 1:
            logger.info(
                f"[{i+1}/{len(workloads)}] B={B} mp={max_num_pages} prog={num_programs} "
                f"sum_sl={sum_sl} util={util:.3f} "
                f"ret={early_return_frac:.2f} contig={contig_frac_overall:.2f}"
            )

    return {"results": results}


@app.local_entrypoint()
def main():
    import json
    out = inspect_all.remote()
    results = out["results"]

    out_path = PROJECT_ROOT / "experiments" / "workload_profile_raw.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Raw results -> {out_path}")
    print(f"Total workloads inspected: {len(results)}")
