import torch


@torch.inference_mode()
def capture_prefill(model, input_ids):
    static_input = input_ids.clone()
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            model(static_input)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = model(static_input)
    return graph, static_input, static_output


class DecodeGraph:
    @torch.inference_mode()
    def __init__(self, model, kv_cache, use_triton_attention=False):
        device = model.transformer.wte.weight.device
        if device.type != "cuda" or torch.version.hip is not None:
            raise ValueError("decode graphs require an NVIDIA CUDA GPU")
        if kv_cache.keys.ndim != 5:
            raise ValueError("KV cache must have five dimensions")
        batch_size = kv_cache.keys.shape[1]
        self.capacity = model._validate_kv_cache(kv_cache, batch_size)
        if not 0 < kv_cache.length < self.capacity:
            raise ValueError("decode requires a filled prompt and room for a new token")
        if use_triton_attention and kv_cache.keys.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Triton decode attention requires FP16 or BF16")
        if use_triton_attention and not (
            kv_cache.keys.is_contiguous() and kv_cache.values.is_contiguous()
        ):
            raise ValueError("Triton decode attention requires contiguous cache buffers")

        self.model = model
        self.kv_cache = kv_cache
        self.length = kv_cache.length
        self.static_input = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        self.position = torch.full((1,), self.length, dtype=torch.long, device=device)
        self.cache_positions = torch.arange(kv_cache.keys.shape[3], device=device).view(1, -1)
        kv_cache.keys[:, :, :, self.length :].zero_()
        kv_cache.values[:, :, :, self.length :].zero_()

        def step():
            attention_mask = None if use_triton_attention else self.cache_positions <= self.position
            return model._decode_at_position(
                self.static_input, kv_cache, self.position, attention_mask
            )

        with torch.cuda.device(device):
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(3):
                    step()
            torch.cuda.current_stream().wait_stream(warmup_stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = step()
                self.position.add_(1)
            self.position.fill_(self.length)

    @torch.inference_mode()
    def decode(self, input_ids):
        if (
            input_ids.shape != self.static_input.shape
            or input_ids.device != self.static_input.device
            or input_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("decode expects one integer token per sequence on the captured device")
        if self.length != self.kv_cache.length:
            raise ValueError("cache length changed outside this decode graph")
        if self.length >= self.capacity:
            raise ValueError("sequence exceeds KV cache capacity or block_size")

        with torch.cuda.device(self.static_input.device):
            self.static_input.copy_(input_ids)
            self.graph.replay()
        self.length += 1
        self.kv_cache.length = self.length
        return self.output
