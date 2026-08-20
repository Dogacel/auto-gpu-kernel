# Experiment 2 — FP8 tensor-core dot (drop bf16 cast)

## Goal

Replace `q_fp8.to(bf16) @ k_fp8.to(bf16)` with a direct FP8 `tl.dot`.

## Rationale

On Blackwell (B200), FP8 tensor cores deliver ~2x throughput of bf16
at the same accumulator width (f32). The kernel is likely
compute-bound for the larger workloads (seq_len ~ 10K–100K). Dropping
the bf16 cast:

- Halves the input bitwidth feeding the tensor cores (same dot shape).
- Removes two in-kernel conversion ops (fp8 → bf16 on Q and K tiles).

## Risk

- Not 100% sure Triton's `tl.dot` accepts fp8 inputs on this container.
  If it rejects them, the kernel will fail to compile and we revert.
- FP8 dot may have slight accuracy differences vs bf16 intermediate.
  Baseline matched exactly (top-K indices identical), so we have
  some slack; hopefully still within spec.

## Change

Only the in-kernel math path:

```diff
- q_bf = q_fp8.to(tl.bfloat16)
- k_bf = k_fp8.to(tl.bfloat16)
- scores = tl.dot(q_bf, tl.trans(k_bf), out_dtype=tl.float32)
+ scores = tl.dot(q_fp8, tl.trans(k_fp8), out_dtype=tl.float32)
```

Everything else (grid, scales, weights, masking, top-K) unchanged.

## Success criterion

- Correctness still within spec (abs_err ≤ 0.02).
- Measurable speedup on the large-workload side (where compute
  dominates). Mean latency ↓ vs exp_1.
