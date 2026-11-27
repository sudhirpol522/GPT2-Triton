import torch
import triton
import triton.language as tl


@triton.jit
def _embedding_kernel(
    token_ids_ptr,
    token_embeddings_ptr,
    position_embeddings_ptr,
    output_ptr,
    num_elements: tl.constexpr,
    sequence_length: tl.constexpr,
    hidden_size: tl.constexpr,
    block_size: tl.constexpr,
):
    element_offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    valid_elements = element_offsets < num_elements
    token_offsets = element_offsets // hidden_size
    hidden_offsets = element_offsets % hidden_size
    token_ids = tl.load(token_ids_ptr + token_offsets, mask=valid_elements, other=0)
    token_values = tl.load(
        token_embeddings_ptr + token_ids * hidden_size + hidden_offsets,
        mask=valid_elements,
        other=0,
    )
    position_values = tl.load(
        position_embeddings_ptr
        + (token_offsets % sequence_length) * hidden_size
        + hidden_offsets,
        mask=valid_elements,
        other=0,
    )
    tl.store(
        output_ptr + element_offsets,
        token_values.to(tl.float32) + position_values.to(tl.float32),
        mask=valid_elements,
    )


def embedding(token_ids, token_embedding_weight, position_embedding_weight):
    token_ids = token_ids.contiguous()
    batch_size, sequence_length = token_ids.shape
    hidden_size = token_embedding_weight.shape[1]
    output = torch.empty(
        (batch_size, sequence_length, hidden_size),
        device=token_ids.device,
        dtype=token_embedding_weight.dtype,
    )
    _embedding_kernel[(triton.cdiv(output.numel(), 256),)](
        token_ids,
        token_embedding_weight,
        position_embedding_weight,
        output,
        output.numel(),
        sequence_length,
        hidden_size,
        256,
    )
    return output
