# /benchmark

| Arg | Command | When |
|---|---|---|
| `quick` (default) | `modal run scripts/run_modal.py --quick` | Compile + correctness (smallest + largest workload) |
| `stride N` | `modal run scripts/run_modal.py --stride N` | Iteration (default `N=2` → ~10 workloads) |
| `full` | `modal run scripts/run_modal.py` | Lock in final numbers (128 workloads, 10-15 min) |

Pipe output to a file (e.g. `bench.log` in repo root) — `/log-experiment` will attach it.

If Modal crash-loops (container fails to boot repeatedly, not just slow), cancel and diagnose; don't sit waiting.

Report back: pass/fail counts, absolute kernel latency (min / median / max, split small vs large when both are present), max abs/rel error, reference latency.
