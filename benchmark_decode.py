import statistics

import torch
import triton

from cuda_graph import DecodeGraph
from model import GPT2


def make_decode_step(model, prompt, capacity, mode):
    cache = model.create_kv_cache(prompt.shape[0], max_length=capacity)
    model(prompt, kv_cache=cache)
    if mode == "eager":
        return lambda token: model.decode(token, cache), cache
    graph = DecodeGraph(model, cache, use_triton_attention=mode == "triton")
    return graph.decode, cache


def check_decode(model, prompt, tokens, capacity):
    eager, _ = make_decode_step(model, prompt, capacity, "eager")
    max_errors = []
    reference = [eager(token) for token in tokens]
    for mode in ("pytorch", "triton"):
        step, cache = make_decode_step(model, prompt, capacity, mode)
        pointers = cache.keys.data_ptr(), cache.values.data_ptr()
        errors = []
        for expected, token in zip(reference, tokens, strict=True):
            actual = step(token)
            torch.testing.assert_close(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)
            errors.append((actual.float() - expected.float()).abs().max().item())
        assert cache.length == prompt.shape[1] + len(tokens)
        assert pointers == (cache.keys.data_ptr(), cache.values.data_ptr())
        max_errors.append(max(errors))
    return max_errors


def measure_decode(model, prompt, tokens, capacity, mode):
    step, _ = make_decode_step(model, prompt, capacity, mode)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for token in tokens:
        step(token)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / len(tokens)


@torch.inference_mode()
def main():
    torch.manual_seed(42)
    model = GPT2().cuda().half().eval()
    workloads = [(1, 32), (1, 128), (1, 512), (1, 1008), (4, 128), (4, 512), (8, 128), (8, 512)]
    modes = ("eager", "pytorch", "triton")
    steps = 16
    rows = []
    print(f"{torch.cuda.get_device_name()} | GPT-2 124M FP16", flush=True)
    print(f"PyTorch {torch.__version__} | Triton {triton.__version__}", flush=True)
    print("Batch  Prompt   Eager ms   PT graph   Triton graph   PT/Triton   Max error", flush=True)
    for batch_size, prompt_length in workloads:
        capacity = prompt_length + steps
        inputs = torch.randint(model.config.vocab_size, (batch_size, capacity), device="cuda")
        prompt = inputs[:, :prompt_length].contiguous()
        tokens = tuple(token[:, None] for token in inputs[:, prompt_length:].unbind(1))
        errors = check_decode(model, prompt, tokens, capacity)
        samples = {mode: [] for mode in modes}
        for repeat in range(3):
            for mode in modes[repeat:] + modes[:repeat]:
                samples[mode].append(measure_decode(model, prompt, tokens, capacity, mode))
        eager, pytorch, triton_graph = (statistics.median(samples[mode]) for mode in modes)
        rows.append((batch_size, prompt_length, eager, pytorch, triton_graph, max(errors)))
        print(
            f"{batch_size:5d} {prompt_length:7d} {eager:10.3f} {pytorch:10.3f} "
            f"{triton_graph:14.3f} {pytorch / triton_graph:10.2f}x {max(errors):11.6f}",
            flush=True,
        )
    print("Both graph paths matched eager at all 128 growing-cache positions.", flush=True)
    return rows


if __name__ == "__main__":
    results = main()
