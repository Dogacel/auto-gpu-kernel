# CLAUDE.md

Autonomous Triton kernel optimization for DSA TopK Indexer (`dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`).

## Non-negotiable rules

- **Stay in Triton.** No CUDA, no language switching. Suspected Triton/Python bugs are almost always something else — investigate before blaming the compiler. Gluon still counts as Triton.
- **Absolute latencies only.** Speedup ratios lie: reference latency swings 20-30% across Modal VMs. It is normal to see (0.00x), only rely on absolute numbers.
- **One optimization per iteration.** Coupled changes misattribute wins. For sub-5% deltas vs previous best, use `scripts/ab_benchmark.py` (paired same-VM) — cross-VM comparison from `summary.md` is noise.
- **No GPU locally.** All compile + benchmark through Modal.
- **Log every experiment** via `/log-experiment`, including failures.
- **Never stop the loop.** Only the user ends optimization.
- **No benchmark gaming.** No memoizing outputs, no iteration-counter tricks, no `--quick`-specific shortcuts. IMPORTANT: No CUDA graph stuff. Cupti only measures CUDA runtime and it is forbidden to cache specific input pointers, CUDA graphs etc for the submission. You can use event streams to debug runtimes.
- **No web search.** You are in an isolated environment, you shouldn't do any web access or search. You only have access to Modal using the `modal` cli.
- **Don't ask anything to the user.** You are designed to work autonomously and user won't answer your questions.

## Commands

| Command | Purpose |
|---|---|
| `/optimize` | Main loop — see `.claude/commands/optimize.md` |
| `/benchmark <quick\|stride N\|full>` | Run on Modal |
| `/log-experiment` | Snapshot kernel + write `result.md` + update index |

## Modal

```bash
modal run scripts/run_modal.py              # full, 128 workloads, 10-15 min
modal run scripts/run_modal.py --stride 8   # ~16 workloads, ~2-3 min — default iteration
modal run scripts/run_modal.py --quick      # 2 workloads (smallest + largest) — correctness only
modal run scripts/ab_benchmark.py::run --a <path>  # paired A/B vs another kernel file
```

If a Modal container crash-loops (fails to boot repeatedly, not just slow), cancel and fix. Don't wait.

## Repo layout

- `solution/triton/indexer_fused.py` — the kernel you edit
- `solution/triton/indexer_baseline.py` — PyTorch reference (read for numerical semantics and cache layout)
- `experiments/exp_N/` — per-experiment: `plan.md?`, `indexer_fused.py`, `result.md`, `bench.log`
- `experiments/summary.md` — master index, one row per experiment
- `experiments/LESSONS.md` — durable cross-experiment findings (append when a lesson recurs)
- `scripts/ab_benchmark.py` — paired A/B harness for coupled-change disambiguation

## External kernels

You can't rely on external kernels from packages `flashinfer` and `deep_gemm`.

## Kernel I/O (fixed shapes: H=64, D=128, TOPK=2048, page_size=64)

Inputs:
- `q_index_fp8 [B, 64, 128]` `float8_e4m3fn`
- `k_index_cache_fp8 [P, 64, 1, 132]` `int8` — FP8 + scales in deep_gemm SOA format (see hazards)
- `weights [B, 64]` `float32`
- `seq_lens [B]` `int32`
- `block_table [B, max_num_pages]` `int32`

Output (DPS, pre-allocated): `topk_indices [B, 2048]` `int32`. Padding slots must be `-1`.

Formula: `final_score[b, t] = sum_h( relu(q[b,h] · K[t,h]) * weights[b,h] )`, then pick top-K tokens per batch.

## Numerical hazards

- `tl.dot` precision: try `tf32x3` first; fall back to `ieee` if `abs_err > ~0.02`.
- **FP8 KV cache is SOA**: per page the bytes are `[fp8_data (64*128 bytes), scales (64*4 bytes)]`, viewed as `[P, 64, 1, 132]`. A strided view like `view[..., :128]` reads into the scale region on some shapes — always extract fp8 + scales to **contiguous** tensors before use. See `indexer_baseline.py::dequant_fp8_kv_cache`.
- `batch_size` varies across workloads (not always 1). Iterate all batch items.
- Topk output padding positions must be `-1` (initialize `topk_indices.fill_(-1)`).

## Git

Commit after each `/log-experiment` that changed `summary.md`. Tag only when the user asks.
