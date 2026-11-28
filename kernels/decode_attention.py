import torch
import triton
import triton.language as tl


@triton.jit
def _decode_attention_kernel(
    query_ptr,
    keys_ptr,
    values_ptr,
    position_ptr,
    output_ptr,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    query_batch_stride: tl.constexpr,
    query_head_stride: tl.constexpr,
    cache_batch_stride: tl.constexpr,
    cache_head_stride: tl.constexpr,
    cache_token_stride: tl.constexpr,
    block_tokens: tl.constexpr,
    block_head_size: tl.constexpr,
    scale: tl.constexpr,
):
    batch_index = tl.program_id(0)
    head_index = tl.program_id(1)
    valid_length = tl.load(position_ptr).to(tl.int32) + 1
    head_offsets = tl.arange(0, block_head_size)
    query_offset = batch_index * query_batch_stride + head_index * query_head_stride
    query = tl.load(query_ptr + query_offset + head_offsets, head_offsets < head_size, other=0).to(
        tl.float32
    )

    cache_offset = batch_index * cache_batch_stride + head_index * cache_head_stride
    running_max = tl.full((), -float("inf"), tl.float32)
    running_sum = tl.full((), 0, tl.float32)
    output = tl.zeros((block_head_size,), tl.float32)

    for token_start in range(0, valid_length, block_tokens):
        token_positions = token_start + tl.arange(0, block_tokens)
        offsets = cache_offset + token_positions[:, None] * cache_token_stride + head_offsets
        valid = (token_positions[:, None] < valid_length) & (head_offsets < head_size)
        keys = tl.load(keys_ptr + offsets, valid, other=0).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * scale
        scores = tl.where(token_positions < valid_length, scores, -float("inf"))
        new_max = tl.maximum(running_max, tl.max(scores, axis=0))
        rescale = tl.exp(running_max - new_max)
        weights = tl.exp(scores - new_max)
        values = tl.load(values_ptr + offsets, valid, other=0).to(tl.float32)
        output = output * rescale + tl.sum(values * weights[:, None], axis=0)
        running_sum = running_sum * rescale + tl.sum(weights, axis=0)
        running_max = new_max

    output_offset = (batch_index * num_heads + head_index) * head_size
    tl.store(
        output_ptr + output_offset + head_offsets, output / running_sum, head_offsets < head_size
    )


def decode_attention(query, keys, values, cache_position):
    batch_size, num_heads, _, head_size = query.shape
    output = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    _decode_attention_kernel[(batch_size, num_heads)](
        query,
        keys,
        values,
        cache_position,
        output,
        num_heads,
        head_size,
        *query.stride()[:2],
        *keys.stride()[:3],
        64,
        triton.next_power_of_2(head_size),
        head_size**-0.5,
        num_warps=4,
    )
    return output
