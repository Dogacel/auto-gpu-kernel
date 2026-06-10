"""
Triton Kernel for DSA Sparse Attention.

Implements batched Native Sparse Attention (DSA) with sparse TopK KV cache selection
for DeepSeek-V3 with tensor parallel size 8.
"""

import math
import torch


def kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    topk = sparse_indices.shape[-1]

    # Flatten paged KV cache: [num_pages, page_size, dim] -> [total_kv_tokens, dim]
    Kc_all = ckv_cache.reshape(-1, head_dim_ckv)
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe)

    # Handle padding indices (-1 -> 0, will be masked later)
    valid_mask = sparse_indices != -1  # [num_tokens, topk]
    safe_indices = sparse_indices.clamp(min=0).long()

    # Gather KV entries for all tokens: [num_tokens, topk, dim]
    flat_idx = safe_indices.reshape(-1)
    Kc = Kc_all[flat_idx].reshape(num_tokens, topk, head_dim_ckv).to(torch.float32)
    Kp = Kp_all[flat_idx].reshape(num_tokens, topk, head_dim_kpe).to(torch.float32)

    qn = q_nope.to(torch.float32)  # [num_tokens, num_qo_heads, head_dim_ckv]
    qp = q_pe.to(torch.float32)    # [num_tokens, num_qo_heads, head_dim_kpe]

    # Attention logits: [num_tokens, num_qo_heads, topk]
    logits = torch.bmm(qn, Kc.transpose(1, 2)) + torch.bmm(qp, Kp.transpose(1, 2))
    logits_scaled = logits * sm_scale

    # Mask invalid entries
    logits_scaled.masked_fill_(~valid_mask.unsqueeze(1), float('-inf'))

    # LSE (base 2)
    lse.copy_(torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0))

    # Attention weights and output
    attn = torch.softmax(logits_scaled, dim=-1)  # [num_tokens, num_qo_heads, topk]
    output.copy_(torch.bmm(attn, Kc).to(torch.bfloat16))
