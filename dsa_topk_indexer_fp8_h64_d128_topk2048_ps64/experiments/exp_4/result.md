# Experiment 4 — 2026-04-16

**Description:** Replaced per-batch Python loop (with `.item()` sync)
by a single batched `torch.topk` + `torch.gather` page remap + masked
write. One fused post-kernel path; zero CPU↔GPU syncs after the score
kernel launch.

## Results
- Pass: 128/128
- Kernel latency (ms): min 0.284 / mean 0.310 / median 0.309 / max 0.342
- Reference latency (ms): n/a (profile_baseline=False)
- Max abs err: 0.00e+00  |  Max rel err: 0.00e+00 (exact match)
- Mode: full (128 workloads)

## A/B vs exp 2 (same-VM, stride 8)
- B wins 16/16
- mean Δ = −2.098 ms (B faster)
- per-workload range: −23.6% to −93.1%
- largest: 5.093 → 0.352 ms (-93.1%)
- smallest: 0.408 → 0.312 ms (-23.6%)

## Delta vs exp 2 (stride 8 mean)
- exp 2 mean = 2.179 ms → exp 4 mean = 0.310 ms (−85.8%, 7× faster)
- max: 4.810 → 0.342 ms (14× faster)
- min: 0.389 → 0.284 ms (−27%)

## Learnings
- **The per-batch Python loop was the dominant cost.** 6+ small kernel
  launches + a CPU↔GPU `.item()` sync per batch item, with batch
  counts up to ~32 in the larger workloads. Summed to ~2 ms of pure
  overhead, dwarfing the actual score kernel.
- **Exp 2's "kernel" was mostly Python loop.** Mean of 2.179 ms was
  ~2.0 ms per-batch overhead + ~0.2 ms actual kernel. Eliminating
  the overhead reveals the score kernel's true cost is ~0.3 ms.
- **Ref-latency reality check:** CLAUDE.md warns cross-VM variance
  is 20-30%, but we see much bigger (>10×) when hitting .item()
  syncs; the overhead is sensitive to CPU/driver latency on the VM
  rather than GPU time.
- **`torch.topk` on padded scores is safe.** Kernel writes −1e30 for
  out-of-bounds positions → they rank last → mask drops them. No
  need for per-batch `:seq_len` slicing.
- **Gather-based remap is cheaper** than 4 small per-batch ops
  (page_idx_per_token indexing, global_page_idx lookup, multiply-add).
  One big gather replaces B × 4 small kernel launches.

## Next candidates
- Score kernel itself is now dominant (~0.3 ms). Axes to try:
  - kernel-side top-K (eliminate scores buffer entirely)
  - tile shape tuning (num_warps, different BLOCK_T)
  - fuse the gather / top-K into the kernel (write sorted output directly)
  - prefetch / pipelining of K tile loads
