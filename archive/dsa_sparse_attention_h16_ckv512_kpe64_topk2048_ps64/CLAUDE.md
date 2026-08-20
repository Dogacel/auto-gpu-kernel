# CLAUDE.md

Autonomous Triton kernel optimization for DSA sparse attention (`dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`).

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
modal run scripts/run_modal.py              # full, 128 workloads, 3-4 min
modal run scripts/run_modal.py --stride 2   # ~10 workloads, ~1-2 min — default iteration
modal run scripts/run_modal.py --quick      # 2 workloads (smallest + largest) — correctness only
modal run scripts/ab_benchmark.py::run --a <path>  # paired A/B vs another kernel file
```

If a Modal container crash-loops (fails to boot repeatedly, not just slow), cancel and fix. Don't wait.

## Repo layout

- `solution/triton/sparse_fused.py` — the kernel you edit
- `solution/triton/sparse_baseline.py` — PyTorch reference (read for numerical semantics)
- `experiments/exp_N/` — per-experiment: `plan.md?`, `sparse_fused.py`, `result.md`, `bench.log`
- `experiments/summary.md` — master index, one row per experiment
- `experiments/LESSONS.md` — durable cross-experiment findings (append when a lesson recurs)
- `scripts/ab_benchmark.py` — paired A/B harness for coupled-change disambiguation

## External kernels

You can't rely on external kernels from packages `flashinfer` and `deep_gemm`.

## Kernel I/O (fixed shapes: H=16, D_ckv=512, D_kpe=64, TOPK=2048, page_size=64)

Inputs: `q_nope [T,16,512] bf16`, `q_pe [T,16,64] bf16`, `ckv_cache [P,64,512] bf16`, `kpe_cache [P,64,64] bf16`, `sparse_indices [T,2048] int32` (−1 = pad), `sm_scale: float`.
Outputs (DPS, pre-allocated): `output [T,16,512] bf16`, `lse [T,16] f32`.

## Numerical hazards

- `tl.dot` precision: try `tf32x3` first; fall back to `ieee` if `abs_err > ~0.02`.
- High precision operants might need f32 accumulator; bf16 accum can blow up over too many terms.
- LSE expected in **base-2** (`/ log(2)`), not base-e.

## Git

Commit after each `/log-experiment` that changed `summary.md`. Tag only when the user asks.
