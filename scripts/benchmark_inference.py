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

    print("\n--- Results ---")
    print(f"Avg Latency: {avg_latency:.4f}s")
    print(f"Avg TPS: {avg_tps:.2f} tokens/s")

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
