import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from triton.testing import do_bench


@triton.autotune(
    configs=[
        triton.Config(
            {
                "block_rows": block_rows,
                "block_output_features": block_output_features,
                "block_input_features": block_input_features,
            },
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_rows, block_output_features, block_input_features, num_warps, num_stages in [
            (16, 64, 32, 4, 2),
            (32, 64, 32, 4, 3),
            (32, 128, 32, 4, 3),
            (64, 64, 32, 4, 3),
            (64, 128, 32, 4, 4),
            (64, 128, 64, 4, 3),
            (64, 256, 32, 8, 3),
            (128, 128, 32, 8, 3),
        ]
    ],
    key=[
        "num_rows",
        "output_features",
        "input_features",
        "has_bias",
        "apply_gelu",
        "has_residual",
    ],
    do_bench=lambda kernel_call, quantiles: do_bench(
        kernel_call, warmup=5, rep=20, quantiles=quantiles
    ),
)
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


_selected_configs = {}


def linear(inputs, weight, bias=None, gelu=False, residual=None):
    inputs = inputs.contiguous()
    input_features = inputs.shape[-1]
    num_rows = inputs.numel() // input_features
    output_features = weight.shape[0]
    output = torch.empty(
        (*inputs.shape[:-1], output_features), device=inputs.device, dtype=inputs.dtype
    )

    arguments = (
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
    )
    config_key = (
        inputs.device,
        inputs.dtype,
        num_rows,
        output_features,
        input_features,
        bias is not None,
        gelu,
        residual is not None,
    )
    selected_config = _selected_configs.get(config_key)
    if selected_config is None:

        def grid(meta):
            return (
                triton.cdiv(num_rows, meta["block_rows"]),
                triton.cdiv(output_features, meta["block_output_features"]),
            )

        _linear_kernel[grid](*arguments)
        best_config = _linear_kernel.best_config
        block_rows = best_config.kwargs["block_rows"]
        block_output_features = best_config.kwargs["block_output_features"]
        block_input_features = best_config.kwargs["block_input_features"]
        launch_grid = (
            triton.cdiv(num_rows, block_rows),
            triton.cdiv(output_features, block_output_features),
        )
        _selected_configs[config_key] = (
            launch_grid,
            block_rows,
            block_output_features,
            block_input_features,
            best_config.num_warps,
            best_config.num_stages,
        )
    else:
        (
            launch_grid,
            block_rows,
            block_output_features,
            block_input_features,
            num_warps,
            num_stages,
        ) = selected_config
        _linear_kernel.fn[launch_grid](
            *arguments,
            block_rows,
            block_output_features,
            block_input_features,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return output
