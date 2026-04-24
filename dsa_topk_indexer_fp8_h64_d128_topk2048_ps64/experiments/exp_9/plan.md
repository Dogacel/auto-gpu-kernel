# Experiment 9 — Early return from inactive score_kernel programs

## Goal

Skip compute for score_kernel programs where `token_start >= seq_len`
(entire tile is beyond the batch's effective sequence length). Write
-1e30 and return immediately.

## Observation

Current grid is `(B, max_num_pages)`. `max_num_pages` is dimensioned
for the largest batch across workloads — many batches have `seq_len`
far smaller than `max_num_pages * page_size`.

Profile hotspot estimate for a876010b (B=29, max_num_pages=89, sum_sl=8812):
- Total programs: 29 × 89 = 2581
- Active programs (at least one in-bounds token): ceil(sum_sl/64) ≈ 138
- Inactive programs: 2443 (94.7%!)

Each inactive program currently:
- Loads seq_lens, block_table
- Loads Q tile (64×128 FP8 = 8192 bytes)
- Loads K tile (64×128 FP8 from page 0 — masked to garbage)
- Does tl.dot (64×128 @ 128×64 FP8 → 64×64 F32)
- Loads scales (×256 bytes)
- Loads weights (×256 bytes)
- tl.where + sum + tl.where with -1e30
- Stores -1e30 ×64

All wasted. Early-return kills ~90% of score_kernel cost on large
workloads.

## Approach

At top of score_kernel:
```python
seq_len = tl.load(seq_lens_ptr + pid_b)
token_start = pid_p * BLOCK_T

if token_start >= seq_len:
    t_offs = tl.arange(0, BLOCK_T)
    score_off = pid_b * stride_sb + (token_start + t_offs) * stride_st
    tl.store(scores_ptr + score_off, tl.full([BLOCK_T], -1e30, tl.float32))
    return

# Normal path continues...
```

- The early-return branch is **block-uniform**: `seq_len` is the same
  for all threads in the block, so the branch doesn't diverge.
- Still writes -1e30 to keep the scores buffer valid (torch.topk must
  ignore these positions). This preserves exact-match correctness.
- The normal path handles the *partial* case (some tokens in bounds,
  some not) via `tl.where(in_bounds, ...)` as before.

## Risks

1. **`return` at block level in Triton**: needs to be supported by the
   Triton version on Modal. If not, use an if/else branch structure
   instead.
2. **Correctness**: -1e30 sentinel is preserved; topk picks from valid
   scores only. Should be bitwise identical to exp 7.
3. **Measurement:** Large workloads should see big savings (30-40 µs?).
   Small workloads (all-active) pay a ~1-cycle branch overhead, no
   real effect.

## Success criterion

- A/B vs exp 7 on stride 8: B wins ≥12/16, mean Δ ≤ −10%.
- Target: mean 0.049 → 0.040 ms (~20% speedup), concentrated on large.
- Correctness: matched_ratio = 1.0 on all 128 workloads.

## Why not tried before

No prior exp attacked inactive-tile compute. Exp 3 tried `PAGES_PER_PROGRAM > 1`
(loop multiple pages) — opposite direction and regressed. This exp
goes the other way: each program still handles one page, but skips the
page entirely when it's out of range.
