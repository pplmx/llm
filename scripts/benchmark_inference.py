import math
import statistics
import time

import typer

from llm.serving.batch_engine import ContinuousBatchingEngine
from llm.serving.config import ServingConfig
from llm.serving.loader import load_model_and_tokenizer
from llm.serving.schemas import GenerationRequest

app = typer.Typer(pretty_exceptions_show_locals=False)


def run_benchmark(
    engine: ContinuousBatchingEngine,
    tokenizer: object,
    prompt: str,
    max_new_tokens: int,
    num_runs: int,
    warmup: int = 1,
):
    """运行推理基准测试.

    ``tokens/s`` counts TOKENS, via the tokenizer the served model actually
    uses. An earlier revision inferred it from string length::

        num_tokens = max(len(output) - len(prompt), 1)

    That is a character count. It is accidentally right for
    ``TokenizerFactory.from_printable_corpus`` (the demo fallback encodes one
    printable character per token, verified ``len(ids) == len(text)``), but the
    serving path only uses that fallback when ``ServingConfig.model_path`` is
    unset. With a real checkpoint — every ``serve_pretrained.yaml`` style setup —
    the tokenizer is BPE, where a token averages several characters, so the
    reported figure overstates throughput by exactly the chars-per-token ratio
    and two runs are not comparable when the ratio differs (different prompts,
    other tokenizers, other languages). Benchmarking against a
    mis-scaled unit is worse than not benchmarking: the number looks precise.
    """
    print("--- Starting Benchmark ---")
    print(f"Prompt: {prompt}")
    print(f"Max New Tokens: {max_new_tokens}")
    print(f"Num Runs: {num_runs} (Warmup: {warmup})")

    def generate() -> str:
        request = GenerationRequest(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
        )
        return engine.generate_request(request)

    # Warmup
    print("Warming up...")
    for _ in range(warmup):
        generate()

    latencies = []
    tokens_per_second = []

    print("Benchmarking...")
    for i in range(num_runs):
        start_time = time.perf_counter()
        output = generate()
        end_time = time.perf_counter()

        latency = end_time - start_time
        generated = output[len(prompt) :]
        ids = tokenizer.encode(generated)
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        num_tokens = max(len(ids), 1)
        tps = num_tokens / latency

        latencies.append(latency)
        tokens_per_second.append(tps)
        print(f"Run {i + 1}: {latency:.4f}s, {tps:.2f} tokens/s")

    avg_latency = statistics.mean(latencies)
    avg_tps = statistics.mean(tokens_per_second)

    # Latency TAILS, not just the mean. A mean hides the runs that matter for a
    # serving decision: a p99 twice the mean is a different product from a tight
    # distribution with the same average. ROADMAP listed this as the remaining
    # gap — latency had only an end-to-end mean, no percentiles.
    #
    # Percentile method: NEAREST-RANK, so a reported figure is always a latency
    # that actually occurred. ``statistics.quantiles`` interpolates and can
    # return values OUTSIDE the sample range — measured on
    # ``[1.0…1.8, 10.0]`` it gave p95 = 13.69 and p99 = 17.30 from a sample whose
    # maximum is 10.0. A percentile above the slowest observed run is not a
    # measurement of anything.
    def _nearest_rank(values: list[float], pct: float) -> float:
        ordered = sorted(values)
        index = max(0, math.ceil(pct / 100 * len(ordered)) - 1)
        return ordered[index]

    p50_latency = statistics.median(latencies)
    p95_latency = _nearest_rank(latencies, 95)
    p99_latency = _nearest_rank(latencies, 99)
    worst_tps = min(tokens_per_second)

    print("\n--- Results ---")
    print(f"Avg Latency: {avg_latency:.4f}s")
    print(f"P50 Latency: {p50_latency:.4f}s")
    print(f"P95 Latency: {p95_latency:.4f}s")
    print(f"P99 Latency: {p99_latency:.4f}s")
    print(f"Avg TPS: {avg_tps:.2f} tokens/s")
    print(f"Min TPS: {worst_tps:.2f} tokens/s  (the slowest run; lower is worse)")

    return avg_latency, avg_tps


@app.command()
def main(
    prompt: str = typer.Option("Hello, world", help="Input prompt"),
    max_new_tokens: int = typer.Option(50, help="Number of tokens to generate"),
    runs: int = typer.Option(5, help="Number of benchmark runs"),
    device: str = typer.Option("auto", help="Device to use"),
    compile: bool = typer.Option(False, help="Enable torch.compile (if supported in engine)"),
    attn_impl: str = typer.Option(
        "mha",
        help=(
            "Attention implementation to benchmark (a key of ATTENTION_REGISTRY: "
            "mha / mla / flash_attn when installed). Run the command once per "
            "implementation to compare; ROADMAP §10.1 tracked the absence of exactly "
            "this comparison."
        ),
    ),
):
    """
    LLM Inference Benchmark.
    """
    config = ServingConfig(device=device, attn_impl=attn_impl)
    config.compile_model = compile

    model, tokenizer = load_model_and_tokenizer(config)
    engine = ContinuousBatchingEngine.from_serving_config(config, model, tokenizer)

    print(f"Attention impl: {attn_impl}")
    try:
        run_benchmark(engine, tokenizer, prompt, max_new_tokens, runs)
    finally:
        engine.unload_model()


if __name__ == "__main__":
    app()
