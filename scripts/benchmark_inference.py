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
    attn_impl: list[str] = typer.Option(
        ["mha"],
        help=(
            "Attention implementation(s) to benchmark (keys of ATTENTION_REGISTRY: "
            "mha / mla / flash_attn when installed). Repeat the flag to compare "
            "several in one run, e.g. --attn-impl mha --attn-impl mla: the script "
            "then prints a side-by-side table with relative throughput and latency. "
            "Skipping the repetition and comparing the printed numbers by hand was "
            "ROADMAP §10.1's gap."
        ),
    ),
):
    """
    LLM Inference Benchmark.

    With more than one ``--attn-impl`` the implementations are benchmarked in
    sequence on the SAME prompt / token budget, so the numbers are directly
    comparable, and a summary table is printed at the end.
    """
    results: list[tuple[str, float, float]] = []

    for impl in attn_impl:
        config = ServingConfig(device=device, attn_impl=impl)
        config.compile_model = compile

        try:
            model, tokenizer = load_model_and_tokenizer(config)
        except Exception as exc:  # noqa: BLE001 - a missing backend must not abort the comparison
            # flash_attn is an optional CUDA-only install; the point of a
            # comparison run is to report what is available, not to fail because
            # one requested backend is absent.
            print(f"\n=== {impl}: SKIPPED ({type(exc).__name__}: {exc}) ===")
            continue

        engine = ContinuousBatchingEngine.from_serving_config(config, model, tokenizer)
        print(f"\n=== Attention impl: {impl} ===")
        try:
            avg_latency, avg_tps = run_benchmark(engine, tokenizer, prompt, max_new_tokens, runs)
        finally:
            engine.unload_model()
        results.append((impl, avg_latency, avg_tps))

    if len(results) > 1:
        _print_comparison(results)
    elif not results:
        raise typer.Exit(code=1)


def _print_comparison(results: list[tuple[str, float, float]]) -> None:
    """Print a side-by-side table, with the best values marked.

    Relative columns are computed against the FIRST successful implementation,
    so the table answers "how much faster is B than A" regardless of how many
    entries there are. Ratios are printed rather than only raw numbers because
    absolute tokens/s depends entirely on the host, while the ratio is the
    reproducible part.
    """
    baseline_name, baseline_latency, baseline_tps = results[0]

    print("\n--- Comparison ---")
    header = f"{'impl':<12} {'avg latency':>12} {'avg tok/s':>10} {'speedup':>9} {'latency':>9}"
    print(header)
    print("-" * len(header))
    for name, latency, tps in results:
        speedup = tps / baseline_tps if baseline_tps else float("nan")
        latency_ratio = latency / baseline_latency if baseline_latency else float("nan")
        print(f"{name:<12} {latency:>11.4f}s {tps:>10.2f} {speedup:>8.2f}x {latency_ratio:>8.2f}x")

    best = max(results, key=lambda row: row[2])
    if best[0] != baseline_name:
        print(f"\nFastest: {best[0]} ({best[2]:.2f} tok/s, {best[2] / baseline_tps:.2f}x {baseline_name})")
    else:
        print(f"\nFastest: {baseline_name} (the baseline)")


if __name__ == "__main__":
    app()
