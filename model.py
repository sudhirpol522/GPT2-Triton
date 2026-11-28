from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    layer_norm_epsilon: float = 1e-5


@dataclass
class KVCache:
    keys: torch.Tensor
    values: torch.Tensor
    length: int = 0


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(
        self,
        x,
        use_triton=False,
        residual=None,
        kv_cache=None,
        layer_index=0,
        cache_position=None,
        attention_mask=None,
    ):
        batch, length, width = x.shape
        if use_triton:
            from kernels import causal_attention, linear

            qkv = linear(x, self.c_attn.weight, self.c_attn.bias)
        else:
            qkv = self.c_attn(x)

        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, length, self.n_head, width // self.n_head).transpose(1, 2)
        k = k.view(batch, length, self.n_head, width // self.n_head).transpose(1, 2)
        v = v.view(batch, length, self.n_head, width // self.n_head).transpose(1, 2)

        if use_triton:
            y = causal_attention(q, k, v)
            return linear(y, self.c_proj.weight, self.c_proj.bias, residual=residual)

        if cache_position is not None:
            keys = kv_cache.keys[layer_index]
            values = kv_cache.values[layer_index]
            keys.index_copy_(2, cache_position, k)
            values.index_copy_(2, cache_position, v)
            if attention_mask is None:
                from kernels.decode_attention import decode_attention

                y = decode_attention(q, keys, values, cache_position)
            else:
                y = F.scaled_dot_product_attention(q, keys, values, attn_mask=attention_mask)
        elif kv_cache is not None:
            start = kv_cache.length
            end = start + length
            kv_cache.keys[layer_index, :, :, start:end].copy_(k)
            kv_cache.values[layer_index, :, :, start:end].copy_(v)
            k = kv_cache.keys[layer_index, :, :, :end]
            v = kv_cache.values[layer_index, :, :, :end]
            y = F.scaled_dot_product_attention(q, k, v, is_causal=start == 0)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(batch, length, width)
        y = self.c_proj(y)
        return y if residual is None else y + residual


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x, use_triton=False, residual=None):
        if use_triton:
            from kernels import linear

            x = linear(x, self.c_fc.weight, self.c_fc.bias, gelu=True)
            return linear(x, self.c_proj.weight, self.c_proj.bias, residual=residual)

        x = self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))
        return x if residual is None else x + residual


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def forward(
        self,
        x,
        use_triton=False,
        kv_cache=None,
        layer_index=0,
        cache_position=None,
        attention_mask=None,
    ):
        if use_triton:
            from kernels import layer_norm

            y = layer_norm(x, self.ln_1.weight, self.ln_1.bias, self.ln_1.eps)
            x = self.attn(y, use_triton=True, residual=x)
            y = layer_norm(x, self.ln_2.weight, self.ln_2.bias, self.ln_2.eps)
        else:
            x = self.attn(
                self.ln_1(x),
                residual=x,
                kv_cache=kv_cache,
                layer_index=layer_index,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )
            y = self.ln_2(x)
        return self.mlp(y, use_triton=use_triton, residual=x)


class GPT2(nn.Module):
    def __init__(self, config=None, use_triton=False):
        super().__init__()
        self.config = config or GPT2Config()
        self.use_triton = use_triton
        if self.config.n_embd % self.config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(self.config.vocab_size, self.config.n_embd),
                "wpe": nn.Embedding(self.config.block_size, self.config.n_embd),
                "h": nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)]),
                "ln_f": nn.LayerNorm(self.config.n_embd, eps=self.config.layer_norm_epsilon),
            }
        )
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def create_kv_cache(self, batch_size, max_length=None):
        max_length = self.config.block_size if max_length is None else max_length
        if batch_size <= 0 or not 0 < max_length <= self.config.block_size:
            raise ValueError("batch_size must be positive and max_length must be within block_size")
        shape = (
            self.config.n_layer,
            batch_size,
            self.config.n_head,
            max_length,
            self.config.n_embd // self.config.n_head,
        )
        weight = self.transformer.wte.weight
        return KVCache(weight.new_empty(shape), weight.new_empty(shape))

    def decode(self, input_ids, kv_cache):
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("decode expects one token per sequence")
        if kv_cache.length == 0:
            raise ValueError("fill the KV cache with a prompt before decoding")
        return self(input_ids, kv_cache=kv_cache)

    def _validate_kv_cache(self, kv_cache, batch_size):
        if self.use_triton:
            raise ValueError("KV cache requires the PyTorch model path")
        weight = self.transformer.wte.weight
        if kv_cache.keys.ndim != 5:
            raise ValueError("KV cache must have five dimensions")
        capacity = kv_cache.keys.shape[3]
        expected_shape = (
            self.config.n_layer,
            batch_size,
            self.config.n_head,
            capacity,
            self.config.n_embd // self.config.n_head,
        )
        for buffer in (kv_cache.keys, kv_cache.values):
            if buffer.shape != expected_shape:
                raise ValueError("KV cache shape does not match the model and batch")
            if buffer.device != weight.device or buffer.dtype != weight.dtype:
                raise ValueError("KV cache must use the model's device and dtype")
        return min(capacity, self.config.block_size)

    def _decode_at_position(self, input_ids, kv_cache, cache_position, attention_mask):
        x = self.transformer.wte(input_ids) + self.transformer.wpe(cache_position)
        for layer_index, block in enumerate(self.transformer.h):
            x = block(
                x,
                kv_cache=kv_cache,
                layer_index=layer_index,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )
        return self.lm_head(self.transformer.ln_f(x))

    def forward(self, input_ids, kv_cache=None):
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape (batch, sequence) with nonzero sizes")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must contain integer token IDs")
        if input_ids.shape[1] > self.config.block_size:
            raise ValueError("sequence length exceeds block_size")

        start = 0
        if kv_cache is not None:
            if torch.is_grad_enabled():
                raise RuntimeError("KV cache requires torch.no_grad() or torch.inference_mode()")
            capacity = self._validate_kv_cache(kv_cache, input_ids.shape[0])
            start = kv_cache.length
            if start < 0 or start + input_ids.shape[1] > capacity:
                raise ValueError("sequence exceeds KV cache capacity or block_size")
            if start > 0 and input_ids.shape[1] != 1:
                raise ValueError("pass one token per sequence after filling the KV cache")

        if self.use_triton:
            if input_ids.device.type != "cuda" or torch.version.hip is not None:
                raise ValueError("Triton forward requires an NVIDIA CUDA GPU")
            if torch.is_grad_enabled():
                raise RuntimeError(
                    "Triton forward requires torch.no_grad() or torch.inference_mode()"
                )
            from kernels import embedding, layer_norm, linear

            x = embedding(input_ids, self.transformer.wte.weight, self.transformer.wpe.weight)
        else:
            positions = torch.arange(start, start + input_ids.shape[1], device=input_ids.device)
            x = self.transformer.wte(input_ids) + self.transformer.wpe(positions)

        for layer_index, block in enumerate(self.transformer.h):
            x = block(x, use_triton=self.use_triton, kv_cache=kv_cache, layer_index=layer_index)

        if self.use_triton:
            norm = self.transformer.ln_f
            x = layer_norm(x, norm.weight, norm.bias, norm.eps)
            return linear(x, self.lm_head.weight)
        logits = self.lm_head(self.transformer.ln_f(x))
        if kv_cache is not None:
            kv_cache.length += input_ids.shape[1]
        return logits
