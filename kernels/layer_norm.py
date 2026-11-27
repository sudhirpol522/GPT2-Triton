import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    hidden_size: tl.constexpr,
    epsilon: tl.constexpr,
    block_size: tl.constexpr,
):
    row_index = tl.program_id(0)
    hidden_offsets = tl.arange(0, block_size)
    inputs = tl.load(
        input_ptr + row_index * hidden_size + hidden_offsets,
        mask=hidden_offsets < hidden_size,
        other=0,
    ).to(tl.float32)
    row_mean = tl.sum(inputs, axis=0) / hidden_size
    centered_inputs = tl.where(hidden_offsets < hidden_size, inputs - row_mean, 0.0)
    row_variance = tl.sum(centered_inputs * centered_inputs, axis=0) / hidden_size
    weight = tl.load(
        weight_ptr + hidden_offsets, mask=hidden_offsets < hidden_size, other=0
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + hidden_offsets, mask=hidden_offsets < hidden_size, other=0
    ).to(tl.float32)
    normalized_output = (
        centered_inputs * tl.rsqrt(row_variance + epsilon) * weight + bias
    )
    tl.store(
        output_ptr + row_index * hidden_size + hidden_offsets,
        normalized_output,
        mask=hidden_offsets < hidden_size,
    )


def layer_norm(inputs, weight, bias, epsilon=1e-5):
    inputs = inputs.contiguous()
    hidden_size = inputs.shape[-1]
    output = torch.empty_like(inputs)
    _layer_norm_kernel[(inputs.numel() // hidden_size,)](
        inputs,
        weight,
        bias,
        output,
        hidden_size,
        epsilon,
        triton.next_power_of_2(hidden_size),
    )
    return output
