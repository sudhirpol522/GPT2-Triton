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
