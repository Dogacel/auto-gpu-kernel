# Experiment 1 — 2026-04-16

**Description:** First Triton kernel. Flash-attention-style fused kernel with online softmax (base 2), one program per query token. All 16 heads share the same TopK KV → processed together as M=16 in tensor cores. Pre-scaled Q (bf16→fp32→bf16) lost 7-bit mantissa precision and caused `abs_err=3e-2`; moved `sm_scale_log2e` multiply post-dot in fp32 and err dropped to 1.56e-02.

Block shapes: `BLOCK_N=64` (matches page_size=64), `D_CKV=512` full, `D_KPE=64` full. Inner dots: `q_nope(16,512) · kcᵀ(512,64)` and `p(16,64) · kc(64,512)`, both `tl.dot` with bf16 inputs + fp32 accumulator. 32 TopK blocks per token.

Launch: `grid=(num_tokens,)`, `num_warps=8`, `num_stages=2`. KV cache flattened via `.view(num_pages*page_size, D)` — no copy.

## Results
- Pass: 12/12
- Kernel latency (ms): small=0.092 / large=0.109 / overall=0.092 (min) / 0.100 (median) / 0.109 (max)
- Reference latency (ms): not profiled (profile_baseline=False)
- Max abs err: 1.56e-02  |  Max rel err: 1.98e+03 (high rel because true output has tiny values; abs_err is the honest metric)
- Mode: stride 2 (12 workloads)

## Learnings
- **Pre-scaling Q by `sm_scale * log2e` and casting bf16 → fp32 → bf16 loses precision** — the mantissa truncation compounds over the 512-element dot × 32 topk blocks. Apply scale post-dot on the fp32 logits instead.
- Max abs err 1.56e-02 is at the limit but under the 0.02 threshold — no need for `input_precision="ieee"` yet. If a future change nudges err higher, switch to `tf32x3` first.
- Latencies split cleanly into small (~0.09ms) and large (~0.11ms) groups — suggests different `num_tokens` between the two halves. Worth running workload-inspector before any per-regime specialization.
- 0.00x speedup printed because `profile_baseline=False` — we can't compare vs reference from summary output, only absolute latencies. Use ab_benchmark for paired comparisons.
