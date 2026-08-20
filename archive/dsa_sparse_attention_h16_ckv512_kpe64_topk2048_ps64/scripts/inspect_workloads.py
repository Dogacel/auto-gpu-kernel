"""
Workload Inspector for dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.

Inspects input tensors across ALL workloads and prints distributions:
- num_tokens (T) distribution
- num_pages (P) range
- sparse_indices padding (-1) count per token
- sparse_indices page-locality (unique pages per token)
- total workload count

Only `sparse_indices` is actually loaded (q/kv are random placeholders);
T and P come from workload.axes, which is cheap metadata.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("flashinfer-workload-inspector")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest", add_python="3.12")
    .apt_install("git")
    .pip_install("huggingface_hub")
    .run_commands(
        "pip install --force-reinstall --upgrade "
        "git+https://github.com/flashinfer-ai/flashinfer-bench.git@main",
    )
)


@app.function(image=image, timeout=900, volumes={TRACE_SET_PATH: trace_volume})
def inspect_workloads() -> dict:
    """Load trace set and compute statistics across all workloads."""
    import numpy as np
    import torch
    from flashinfer_bench import TraceSet

    DEF_NAME = "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64"

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    traces = trace_set.workloads.get(DEF_NAME, [])
    print(f"Total workloads for {DEF_NAME}: {len(traces)}")

    per_workload = []  # one dict per workload
    # Global (per-token) distributions
    all_pad_counts = []
    all_unique_pages = []
    all_valid_counts = []
    all_contig_frac = []

    cross_token_reuse = []  # per-workload mean reuse

    page_sz = 64

    for w_idx, trace in enumerate(traces):
        wl = trace.workload
        T = int(wl.axes.get("num_tokens"))
        P = int(wl.axes.get("num_pages"))

        # Load sparse_indices tensor from safetensors
        si_spec = wl.inputs["sparse_indices"]
        # si_spec is a SafetensorsInput with .path (relative to root) and .tensor_key
        st_path = Path(trace_set.root) / si_spec.path
        from safetensors import safe_open
        with safe_open(str(st_path), framework="numpy") as f:
            si = f.get_tensor(si_spec.tensor_key)
        si = np.asarray(si)
        TOPK = si.shape[-1]

        if si.shape[0] != T:
            print(f"WARN workload[{w_idx}] axes.num_tokens={T} but sparse_indices.shape[0]={si.shape[0]}")
            T = si.shape[0]

        # Per-token statistics
        pad_per_token = (si == -1).sum(axis=-1)
        valid_per_token = TOPK - pad_per_token
        all_pad_counts.extend(pad_per_token.tolist())
        all_valid_counts.extend(valid_per_token.tolist())

        pages_per_token = []
        contig_per_token = []
        for t in range(T):
            row = si[t]
            valid_mask = row != -1
            valid = row[valid_mask]
            if valid.size == 0:
                pages_per_token.append(0)
                contig_per_token.append(0.0)
                continue
            pg = valid // page_sz
            pages_per_token.append(int(np.unique(pg).size))
            if valid.size >= 2:
                diffs = np.diff(valid)
                contig_per_token.append(float((diffs == 1).mean()))
            else:
                contig_per_token.append(0.0)
        all_unique_pages.extend(pages_per_token)
        all_contig_frac.extend(contig_per_token)

        # Cross-token page reuse
        page_token_sets = {}
        for t in range(T):
            row = si[t]
            valid = row[row != -1]
            if valid.size == 0:
                continue
            pgs = np.unique(valid // page_sz)
            for pg in pgs.tolist():
                page_token_sets.setdefault(pg, set()).add(t)
        reuse = float(np.mean([len(s) for s in page_token_sets.values()])) if page_token_sets else 0.0
        cross_token_reuse.append(reuse)

        per_workload.append({
            "idx": w_idx,
            "uuid": wl.uuid[:8],
            "T": T,
            "P": P,
            "TOPK": int(TOPK),
            "pad_p50": float(np.median(pad_per_token)),
            "pad_max": int(pad_per_token.max()),
            "pad_min": int(pad_per_token.min()),
            "valid_p50": float(np.median(valid_per_token)),
            "unique_pages_p50": float(np.median(pages_per_token)),
            "unique_pages_max": int(max(pages_per_token)),
            "unique_pages_min": int(min(pages_per_token)),
            "contig_frac_mean": float(np.mean(contig_per_token)),
            "cross_token_page_reuse_mean": reuse,
        })

        print(f"workload[{w_idx:2d}] T={T:5d} P={P:5d} "
              f"pad: min={pad_per_token.min():4d} p50={np.median(pad_per_token):.0f} max={pad_per_token.max():4d} | "
              f"uniq_pg: p50={np.median(pages_per_token):.0f} max={max(pages_per_token)} | "
              f"contig={np.mean(contig_per_token):.3f} "
              f"reuse={reuse:.2f}")

    def pct(arr, p):
        return float(np.percentile(arr, p)) if len(arr) else 0.0

    def summarize(name, arr):
        if not len(arr):
            return f"{name}: empty"
        arr = np.asarray(arr)
        return (f"{name}: min={arr.min():.2f} p10={pct(arr,10):.2f} "
                f"p50={pct(arr,50):.2f} p90={pct(arr,90):.2f} max={arr.max():.2f} "
                f"mean={arr.mean():.2f} n={len(arr)}")

    Ts = [w["T"] for w in per_workload]
    Ps = [w["P"] for w in per_workload]

    # Histogram over T, bucketed
    bucket_edges = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    t_hist = {}
    for t in Ts:
        placed = False
        for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:]):
            if lo <= t < hi:
                key = f"[{lo},{hi})"
                t_hist[key] = t_hist.get(key, 0) + 1
                placed = True
                break
        if not placed:
            key = f">={bucket_edges[-1]}"
            t_hist[key] = t_hist.get(key, 0) + 1

    print()
    print("=== SUMMARY ===")
    print(f"total_workloads: {len(traces)}")
    print(summarize("T (num_tokens)", Ts))
    print(f"T histogram: {t_hist}")
    print(f"T values sorted: {sorted(Ts)}")
    print(f"T exact counts: "
          f"{sorted([(t, Ts.count(t)) for t in set(Ts)], key=lambda x: -x[1])}")
    print(summarize("P (num_pages)", Ps))
    print(f"P values sorted: {sorted(Ps)}")
    print(summarize("pad_per_token", all_pad_counts))
    print(summarize("valid_per_token", all_valid_counts))
    print(summarize("unique_pages_per_token", all_unique_pages))
    print(summarize("contig_frac_per_token", all_contig_frac))
    print(summarize("cross_token_page_reuse_per_workload", cross_token_reuse))

    pad_arr = np.asarray(all_pad_counts)
    up = np.asarray(all_unique_pages)
    n_tokens = len(pad_arr)
    frac_any_pad = float((pad_arr > 0).mean())
    frac_all_valid = float((pad_arr == 0).mean())
    frac_mostly_invalid = float((pad_arr > 1024).mean())
    frac_half_invalid = float((pad_arr > TOPK // 2).mean() if 'TOPK' in dir() else 0.0)
    print(f"fraction_tokens_any_padding: {frac_any_pad:.3f}")
    print(f"fraction_tokens_fully_valid (pad==0): {frac_all_valid:.3f}")
    print(f"fraction_tokens_mostly_invalid (pad>1024): {frac_mostly_invalid:.3f}")

    return {
        "total_workloads": len(traces),
        "per_workload": per_workload,
        "Ts": Ts,
        "Ps": Ps,
        "T_hist": t_hist,
        "pad_dist": {
            "min": int(pad_arr.min()),
            "p10": pct(all_pad_counts, 10),
            "p50": pct(all_pad_counts, 50),
            "p90": pct(all_pad_counts, 90),
            "max": int(pad_arr.max()),
            "mean": float(pad_arr.mean()),
            "frac_any_pad": frac_any_pad,
            "frac_all_valid": frac_all_valid,
            "frac_mostly_invalid": frac_mostly_invalid,
        },
        "unique_pages_dist": {
            "min": int(up.min()),
            "p10": pct(all_unique_pages, 10),
            "p50": pct(all_unique_pages, 50),
            "p90": pct(all_unique_pages, 90),
            "max": int(up.max()),
            "mean": float(up.mean()),
        },
        "contig_dist": {
            "p10": pct(all_contig_frac, 10),
            "p50": pct(all_contig_frac, 50),
            "p90": pct(all_contig_frac, 90),
            "mean": float(np.mean(all_contig_frac)),
        },
        "cross_reuse_dist": {
            "p10": pct(cross_token_reuse, 10),
            "p50": pct(cross_token_reuse, 50),
            "p90": pct(cross_token_reuse, 90),
            "mean": float(np.mean(cross_token_reuse)),
        },
    }


@app.local_entrypoint()
def main():
    result = inspect_workloads.remote()
    import json
    brief = {k: v for k, v in result.items() if k != "per_workload"}
    brief["per_workload_head"] = result["per_workload"][:5]
    brief["per_workload_tail"] = result["per_workload"][-5:]
    brief["num_per_workload_entries"] = len(result["per_workload"])
    print(json.dumps(brief, indent=2))
    # Also persist full
    Path("/tmp/claude-1000/modal_logs").mkdir(parents=True, exist_ok=True)
    out = Path("/tmp/claude-1000/modal_logs/full_result.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"\nFull result dumped to {out}")
