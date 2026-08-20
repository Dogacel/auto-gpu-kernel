# Experiment 4 — Batched top-K + gather remap (kill Python per-batch loop)

## Goal

Replace the `for b in range(batch_size)` Python loop with fully-batched
tensor ops: one `torch.topk` over `[B, max_scored]`, one `gather` for
page remap, one masked write. Eliminates per-batch `.item()` sync and
per-batch launch overhead.

## Rationale

Current post-kernel path:

```python
for b in range(batch_size):
    seq_len = int(seq_lens[b].item())   # CPU/GPU sync per batch
    if seq_len == 0: continue
    actual_topk = min(topk, seq_len)
    seq_scores = scores[b, :seq_len]
    _, topk_idx = torch.topk(seq_scores, actual_topk)  # small topk per batch
    # + 4 small element-wise / gather ops
    topk_indices[b, :actual_topk] = topk_tokens.to(torch.int32)
```

For each batch item this is ~6 small kernel launches + 1 CPU sync. With
B up to ~64 in large workloads, that's ~384 launches and ~64 syncs
after the big score kernel. Each launch ~5 µs on B200, so this alone
could be ~2 ms — possibly the whole "why exp 2 is 2.2 ms" overhead.

Kernel itself already writes `-1e30` for out-of-bounds positions, so a
**single** `torch.topk(scores, topk, dim=-1)` works correctly: invalid
positions score below everything real and end up in positions beyond
`actual_topk[b]` — we just mask those to `-1`.

## Implementation

Replace the Python loop with:

```python
# scores: [B, max_scored], padding = -1e30
# Single top-K call. Always pick topk=2048 per batch.
_, topk_idx = torch.topk(scores, topk, dim=-1)   # [B, topk], int64

page_idx_per_token = (topk_idx // page_size).clamp_(max=max_num_pages - 1)  # [B, topk]
offset_per_token   = topk_idx % page_size                                    # [B, topk]

# Gather global page_id from block_table per (b, token): block_table[b, page_idx_per_token[b,i]]
global_page_idx = torch.gather(block_table.long(), 1, page_idx_per_token)    # [B, topk]

topk_tokens = (global_page_idx * page_size + offset_per_token).to(torch.int32)  # [B, topk]

# Mask positions beyond actual seq_len to -1
actual_topks = torch.minimum(seq_lens.to(torch.long), torch.full_like(seq_lens.to(torch.long), topk))
arange = torch.arange(topk, device=device).unsqueeze(0)                       # [1, topk]
mask = arange < actual_topks.unsqueeze(-1)                                    # [B, topk]
topk_indices.copy_(torch.where(mask, topk_tokens, torch.full_like(topk_tokens, -1)))
```

All GPU-resident. No `.item()`. No Python loop.

## Risk

- `clamp_` on `page_idx_per_token` — if a score at an out-of-bounds
  position somehow scored well (shouldn't, since we wrote -1e30),
  the gather index might exceed `max_num_pages - 1` and crash. Clamp
  is defensive; mask drops the result anyway.
- Correctness sensitive to tie-breaking. `torch.topk` isn't guaranteed
  to match per-batch behavior at ties, but task spec is top-K tokens
  with matched_ratio ≥ spec — not exact index match.
- Allocation cost: `topk_idx` ([B, topk]) is ~B × 16 KB. For B=64
  that's 1 MB — trivial.

## Success criterion

- `/benchmark quick` passes correctness.
- `/benchmark stride 8` shows measurable improvement. Target: ≥5% mean
  latency reduction, or clear improvement on large workloads (where
  per-batch overhead multiplies).
- A/B vs exp 2 same-VM if result is <5%.

## Alternatives considered

- Kernel-side top-K (reduction tree in shared memory): much more
  complex and memory-heavy; defer.
- Fuse page remap into score kernel's output: harder; kernel would
  need `block_table` globally readable and output would need different
  layout. Defer.
