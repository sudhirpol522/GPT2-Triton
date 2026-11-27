import torch
import triton
import triton.language as tl


@triton.jit
def _causal_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    output_ptr,
    sequence_length: tl.constexpr,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    query_batch_stride: tl.constexpr,
    query_head_stride: tl.constexpr,
    query_sequence_stride: tl.constexpr,
    key_batch_stride: tl.constexpr,
    key_head_stride: tl.constexpr,
    key_sequence_stride: tl.constexpr,
    value_batch_stride: tl.constexpr,
    value_head_stride: tl.constexpr,
    value_sequence_stride: tl.constexpr,
    attention_scale: tl.constexpr,
    block_queries: tl.constexpr,
    block_keys: tl.constexpr,
    block_head_size: tl.constexpr,
):
    query_block_index = tl.program_id(0)
    batch_head_index = tl.program_id(1)
    batch_index = batch_head_index // num_heads
    head_index = batch_head_index % num_heads
    query_positions = query_block_index * block_queries + tl.arange(0, block_queries)
    key_block_offsets = tl.arange(0, block_keys)
    head_offsets = tl.arange(0, block_head_size)

    query = tl.load(
        query_ptr
        + batch_index * query_batch_stride
        + head_index * query_head_stride
        + query_positions[:, None] * query_sequence_stride
        + head_offsets[None, :],
        mask=(query_positions[:, None] < sequence_length)
        & (head_offsets[None, :] < head_size),
        other=0,
    )
    running_max = tl.full((block_queries,), -float("inf"), tl.float32)
    running_sum = tl.zeros((block_queries,), tl.float32)
    output_accumulator = tl.zeros((block_queries, block_head_size), tl.float32)

    for key_block_start in range(
        0, (query_block_index + 1) * block_queries, block_keys
    ):
        key_positions = key_block_start + key_block_offsets
        key = tl.load(
            key_ptr
            + batch_index * key_batch_stride
            + head_index * key_head_stride
            + key_positions[None, :] * key_sequence_stride
            + head_offsets[:, None],
            mask=(key_positions[None, :] < sequence_length)
            & (head_offsets[:, None] < head_size),
            other=0,
        )
        attention_scores = tl.dot(query, key, input_precision="ieee") * attention_scale
        attention_scores = tl.where(
            (key_positions[None, :] < sequence_length)
            & (query_positions[:, None] >= key_positions[None, :]),
            attention_scores,
            -float("inf"),
        )
        updated_max = tl.maximum(running_max, tl.max(attention_scores, axis=1))
        attention_weights = tl.exp(attention_scores - updated_max[:, None])
        rescale_factor = tl.exp(running_max - updated_max)
        running_sum = running_sum * rescale_factor + tl.sum(attention_weights, axis=1)
        output_accumulator = output_accumulator * rescale_factor[:, None]
        value = tl.load(
            value_ptr
            + batch_index * value_batch_stride
            + head_index * value_head_stride
            + key_positions[:, None] * value_sequence_stride
            + head_offsets[None, :],
            mask=(key_positions[:, None] < sequence_length)
            & (head_offsets[None, :] < head_size),
            other=0,
        )
        output_accumulator = tl.dot(
            attention_weights.to(value.dtype),
            value,
            output_accumulator,
            input_precision="ieee",
        )
        running_max = updated_max

    attention_output = output_accumulator / running_sum[:, None]
    output_offsets = (
        (batch_index * sequence_length + query_positions[:, None]) * num_heads
        + head_index
    ) * head_size + head_offsets[None, :]
    tl.store(
        output_ptr + output_offsets,
        attention_output,
        mask=(query_positions[:, None] < sequence_length)
        & (head_offsets[None, :] < head_size),
    )


def causal_attention(query, key, value):
    batch_size, num_heads, sequence_length, head_size = query.shape
    output = torch.empty(
        (batch_size, sequence_length, num_heads, head_size),
        device=query.device,
        dtype=query.dtype,
    )
    _causal_attention_kernel[
        (triton.cdiv(sequence_length, 32), batch_size * num_heads)
    ](
        query,
        key,
        value,
        output,
        sequence_length,
        num_heads,
        head_size,
        *query.stride()[:3],
        *key.stride()[:3],
        *value.stride()[:3],
        head_size**-0.5,
        32,
        32,
        max(16, triton.next_power_of_2(head_size)),
        num_warps=4,
        num_stages=2,
    )
    return output.view(batch_size, sequence_length, num_heads * head_size)
