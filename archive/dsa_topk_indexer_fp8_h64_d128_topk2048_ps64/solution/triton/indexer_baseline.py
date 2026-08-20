"""
DSA TopK Indexer — Baseline (PyTorch reference).

Computes sparse attention scores using ReLU activation and learned weights,
then selects top-K KV cache indices.

Formula: sum(relu(q @ K.T) * weights)

Constants: H=64 (index heads), D=128 (head dim), topk=2048, page_size=64
FP8 quantized inputs (deep_gemm format).
"""

import torch


def dequant_fp8_kv_cache(k_index_cache_fp8):
    """Dequantize FP8 KV cache from deep_gemm format.

    Input: [num_pages, page_size, 1, 132] int8 (interpreted as uint8)
           Memory layout (per page): [fp8_data (page_size * 128 bytes), scales (page_size * 4 bytes)]
    Output: [num_pages, page_size, 128] float32
    """
    k_index_cache_fp8 = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, num_heads, head_dim_sf = k_index_cache_fp8.shape
    head_dim = head_dim_sf - 4  # 128

    kv_flat = k_index_cache_fp8.view(num_pages, page_size * head_dim_sf)

    fp8_bytes = kv_flat[:, :page_size * head_dim].contiguous()
    fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
    fp8_float = fp8_tensor.to(torch.float32)

    scale_bytes = kv_flat[:, page_size * head_dim:].contiguous()
    scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)  # [num_pages, page_size, 1]

    return fp8_float * scale


@torch.no_grad()
def kernel(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table,
           topk_indices):
    """DSA TopK Indexer — baseline PyTorch implementation.

    Args:
        q_index_fp8: [batch_size, 64, 128] float8_e4m3fn
        k_index_cache_fp8: [num_pages, 64, 1, 132] int8
        weights: [batch_size, 64] float32
        seq_lens: [batch_size] int32
        block_table: [batch_size, max_num_pages] int32
        topk_indices: [batch_size, 2048] int32 (output, pre-allocated)
    """
    batch_size = q_index_fp8.shape[0]
    page_size = 64
    index_head_dim = 128
    topk = 2048

    q = q_index_fp8.to(torch.float32)  # [batch, 64, 128]
    K_all = dequant_fp8_kv_cache(k_index_cache_fp8)  # [num_pages, 64, 128]

    topk_indices.fill_(-1)

    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        if seq_len == 0:
            continue

        num_pages_for_seq = (seq_len + page_size - 1) // page_size
        page_indices = block_table[b, :num_pages_for_seq].to(torch.long)

        K_paged = K_all[page_indices]  # [num_pages_for_seq, 64, 128]
        K = K_paged.reshape(-1, index_head_dim)[:seq_len]  # [seq_len, 128]

        q_b = q[b]  # [64, 128]

        scores = q_b @ K.T  # [64, seq_len]
        scores_relu = torch.relu(scores)

        w = weights[b]  # [64]
        weighted_scores = scores_relu * w[:, None]  # [64, seq_len]
        final_scores = weighted_scores.sum(dim=0)  # [seq_len]

        actual_topk = min(topk, seq_len)
        _, topk_idx = torch.topk(final_scores, actual_topk)

        page_idx_per_token = topk_idx // page_size
        offset_per_token = topk_idx % page_size
        global_page_idx = page_indices[page_idx_per_token]
        topk_tokens = global_page_idx * page_size + offset_per_token

        topk_indices[b, :actual_topk] = topk_tokens.to(torch.int32)
