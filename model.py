from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    layer_norm_epsilon: float = 1e-5


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x, use_triton=False, residual=None):
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

    def forward(self, x, use_triton=False):
        if use_triton:
            from kernels import layer_norm

            y = layer_norm(x, self.ln_1.weight, self.ln_1.bias, self.ln_1.eps)
            x = self.attn(y, use_triton=True, residual=x)
            y = layer_norm(x, self.ln_2.weight, self.ln_2.bias, self.ln_2.eps)
        else:
            x = self.attn(self.ln_1(x), residual=x)
            y = self.ln_2(x)
        return self.mlp(y, use_triton=use_triton, residual=x)


class GPT2(nn.Module):
    def __init__(self, config=None, use_triton=False):
        super().__init__()
        self.config = config or GPT2Config()
        self.use_triton = use_triton
        if self.config.n_embd % self.config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(self.config.vocab_size, self.config.n_embd),
            "wpe": nn.Embedding(self.config.block_size, self.config.n_embd),
            "h": nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)]),
            "ln_f": nn.LayerNorm(self.config.n_embd, eps=self.config.layer_norm_epsilon),
        })
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids):
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape (batch, sequence) with nonzero sizes")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must contain integer token IDs")
        if input_ids.shape[1] > self.config.block_size:
            raise ValueError("sequence length exceeds block_size")

        if self.use_triton:
            if input_ids.device.type != "cuda" or torch.version.hip is not None:
                raise ValueError("Triton forward requires an NVIDIA CUDA GPU")
            if torch.is_grad_enabled():
                raise RuntimeError("Triton forward requires torch.no_grad() or torch.inference_mode()")
            from kernels import embedding, layer_norm, linear

            x = embedding(input_ids, self.transformer.wte.weight, self.transformer.wpe.weight)
        else:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            x = self.transformer.wte(input_ids) + self.transformer.wpe(positions)

        for block in self.transformer.h:
            x = block(x, use_triton=self.use_triton)

        if self.use_triton:
            norm = self.transformer.ln_f
            x = layer_norm(x, norm.weight, norm.bias, norm.eps)
            return linear(x, self.lm_head.weight)
        return self.lm_head(self.transformer.ln_f(x))
