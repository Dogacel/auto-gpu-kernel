abandoned: PPP=4 static_range regressed large workloads ~+85% (2.34 → 4.32 ms); PPP=2 runtime loop even worse. Likely register pressure + lost across-program pipelining. Skip-`.contiguous()` ablation was also tried (mixed: small ↓, large ↑); not attributing to exp 3 goal. Reverting to exp 2 kernel. Notable incidental finding: cross-VM variance much larger than docs suggest (saw same exp 2 code at 0.33 ms and 4.0 ms on different VMs).

# Experiment 3 — Pages per program > 1 (reduce grid, amortize Q/W load)

## Goal

Each Triton program handles `PAGES_PER_PROGRAM` pages sequentially
instead of one. Q and per-head weights are loaded **once per program**
and reused across pages.

Grid changes: `(B, max_num_pages)` → `(B, ceil(max_num_pages / PPP))`.

## Rationale

Per-program work in the exp_2 kernel is small:
`tl.dot(64, 128) × (128, 64) → (64, 64)` plus a couple of element-wise
ops. For a big workload (≈1500 pages × B=1), that's 1500 program
launches for a few microseconds of work each. Per-program launch +
SMEM setup + Q fetch overhead adds up.

Processing 4 pages per program:
- Reduces grid size 4×.
- Amortizes `tl.load(Q)` (8 KB fp8) and `tl.load(w)` (256 B f32)
  across 4 dots instead of 1.
- Each program still stays small enough to fit in SMEM (Q is 8 KB,
  K loads rotate per iteration).

## Design notes

- `PAGES_PER_PROGRAM: tl.constexpr = 4`.
- `tl.static_range(PPP)` inside the kernel — unrolls cleanly.
- `if pid_p < max_num_pages:` to guard the tail when `max_num_pages`
  isn't a multiple of PPP.
- K tile, scales, block_table entry loaded per iteration.
- Q and weights hoisted out of the loop.

## Risks

- For small workloads (seq_len ≤ ~200) we may undersubscribe SMs. A
  workload with, say, 3 pages total → 1 program with PPP=4. OK on
  B200 (150 SMs) for tiny workloads that are short anyway.
- If per-page work is already dominated by memory latency (not
  compute), amortizing Q/w loads helps little. Expect a small win on
  small workloads, moderate win on large.

## Alternatives considered (next experiments)

- Full-batch `torch.topk` (eliminates Python per-batch loop).
- Kernel-side top-K (eliminates the scores buffer write/read round trip).
- Strided view (skip `.contiguous()` copies of fp8/scales).

Picked this because it directly reduces launch count, which is the
most plausible remaining overhead given the absolute latencies.
