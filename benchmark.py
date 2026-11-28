import statistics
import time

import torch
import triton

from cuda_graph import capture_prefill
from model import GPT2, GPT2Config


@torch.inference_mode()
def measure_latency(run, *args, repeats=20):
    for _ in range(5):
        run(*args)
    torch.cuda.synchronize()
    samples = []
    for _ in range(3):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            run(*args)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / repeats)
    return statistics.median(samples)


def replay_prefill(graph, static_input, input_ids):
    static_input.copy_(input_ids)
    graph.replay()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark needs an NVIDIA CUDA GPU")
    torch.manual_seed(42)
    config = GPT2Config()
    model = GPT2(config).to(device="cuda", dtype=torch.float16).eval()
    workloads = [
        (1, 1),
        (1, 32),
        (1, 128),
        (1, 512),
        (1, 1024),
        (4, 1),
        (4, 32),
        (4, 128),
        (4, 512),
        (4, 1024),
        (8, 128),
        (8, 512),
    ]
    rows = []
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("PyTorch:", torch.__version__, "Triton:", triton.__version__, flush=True)
    for batch_size, sequence_length in workloads:
        input_ids = torch.randint(
            config.vocab_size, (batch_size, sequence_length), device="cuda"
        )
        with torch.inference_mode():
            model.use_triton = False
            expected = model(input_ids)
            model.use_triton = True
            actual = model(input_ids)
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
            del actual, expected

        times = []
        for use_triton in (False, True):
            model.use_triton = use_triton
            eager_ms = measure_latency(model, input_ids)
            capture_start = time.perf_counter()
            graph, static_input, static_output = capture_prefill(model, input_ids)
            torch.cuda.synchronize()
            capture_ms = (time.perf_counter() - capture_start) * 1000

            with torch.inference_mode():
                for _ in range(2):
                    new_input_ids = torch.randint(
                        config.vocab_size, input_ids.shape, device="cuda"
                    )
                    static_input.copy_(new_input_ids)
                    graph.replay()
                    expected = model(new_input_ids)
                    torch.testing.assert_close(
                        static_output, expected, atol=1e-3, rtol=1e-3
                    )
                    del expected

            graph_ms = measure_latency(replay_prefill, graph, static_input, input_ids)
            times.extend([eager_ms, graph_ms])
            name = "Triton" if use_triton else "PyTorch"
            print(
                f"{(batch_size, sequence_length)} {name}: eager {eager_ms:.3f} ms, graph {graph_ms:.3f} ms, capture+warmup {capture_ms:.1f} ms, checks passed",
                flush=True,
            )
            del graph, static_input, static_output
        rows.append((batch_size, sequence_length, *times))

    print("batch sequence PT_eager PT_graph Triton_eager Triton_graph")
    for (
        batch_size,
        sequence_length,
        pytorch_eager,
        pytorch_graph,
        triton_eager,
        triton_graph,
    ) in rows:
        print(
            f"{batch_size:5d} {sequence_length:8d} {pytorch_eager:8.3f} {pytorch_graph:8.3f} {triton_eager:12.3f} {triton_graph:12.3f}"
        )
    return rows


if __name__ == "__main__":
    results = main()
