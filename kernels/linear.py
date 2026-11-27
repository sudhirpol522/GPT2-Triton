import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    output_ptr,
    num_rows: tl.constexpr,
    output_features: tl.constexpr,
    input_features: tl.constexpr,
    has_bias: tl.constexpr,
    apply_gelu: tl.constexpr,
    has_residual: tl.constexpr,
    block_rows: tl.constexpr,
    block_output_features: tl.constexpr,
    block_input_features: tl.constexpr,
):
    row_offsets = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    output_feature_offsets = tl.program_id(1) * block_output_features + tl.arange(
        0, block_output_features
    )
    input_block_offsets = tl.arange(0, block_input_features)
    accumulator = tl.zeros((block_rows, block_output_features), dtype=tl.float32)

    for input_block_index in range(tl.cdiv(input_features, block_input_features)):
        input_offsets = input_block_index * block_input_features + input_block_offsets
        input_tile = tl.load(
            input_ptr + row_offsets[:, None] * input_features + input_offsets[None, :],
            mask=(row_offsets[:, None] < num_rows)
            & (input_offsets[None, :] < input_features),
            other=0,
        )
        weight_tile = tl.load(
            weight_ptr
            + output_feature_offsets[None, :] * input_features
            + input_offsets[:, None],
            mask=(output_feature_offsets[None, :] < output_features)
            & (input_offsets[:, None] < input_features),
            other=0,
        )
        accumulator = tl.dot(
            input_tile, weight_tile, accumulator, input_precision="ieee"
        )

    if has_bias:
        bias = tl.load(
            bias_ptr + output_feature_offsets,
            mask=output_feature_offsets < output_features,
            other=0,
        ).to(tl.float32)
        accumulator += bias[None, :]

    output_values = accumulator.to(output_ptr.dtype.element_ty).to(tl.float32)
    if apply_gelu:
        gelu_argument = 0.7978845608028654 * (
            output_values + 0.044715 * output_values * output_values * output_values
        )
        output_values = 0.5 * output_values * (1.0 + libdevice.tanh(gelu_argument))

    output_offsets = (
        row_offsets[:, None] * output_features + output_feature_offsets[None, :]
    )
    valid_output = (row_offsets[:, None] < num_rows) & (
        output_feature_offsets[None, :] < output_features
    )
    if has_residual:
        residual = tl.load(
            residual_ptr + output_offsets, mask=valid_output, other=0
        ).to(tl.float32)
        output_values += residual
    tl.store(output_ptr + output_offsets, output_values, mask=valid_output)


def linear(inputs, weight, bias=None, gelu=False, residual=None):
    inputs = inputs.contiguous()
    input_features = inputs.shape[-1]
    num_rows = inputs.numel() // input_features
    output_features = weight.shape[0]
    output = torch.empty(
        (*inputs.shape[:-1], output_features), device=inputs.device, dtype=inputs.dtype
    )
    _linear_kernel[(triton.cdiv(num_rows, 32), triton.cdiv(output_features, 64))](
        inputs,
        weight,
        bias,
        residual,
        output,
        num_rows,
        output_features,
        input_features,
        bias is not None,
        gelu,
        residual is not None,
        32,
        64,
        32,
        num_warps=4,
        num_stages=3,
    )
    return output
