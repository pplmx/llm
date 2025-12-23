import argparse
import statistics
import time

from llm.serving.config import ServingConfig
from llm.serving.engine import LLMEngine


def run_benchmark(engine: LLMEngine, prompt: str, max_new_tokens: int, num_runs: int, warmup: int = 1):
    """运行推理基准测试."""
    print("--- Starting Benchmark ---")
    print(f"Prompt: {prompt}")
    print(f"Max New Tokens: {max_new_tokens}")
    print(f"Num Runs: {num_runs} (Warmup: {warmup})")

    # Warmup
    print("Warming up...")
    for _ in range(warmup):
        engine.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.7)

    latencies = []
    tokens_per_second = []

    print("Benchmarking...")
    for i in range(num_runs):
        start_time = time.perf_counter()
        output = engine.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.7)
        end_time = time.perf_counter()

        latency = end_time - start_time
        num_tokens = len(output)  # Simple char tokenizer approximation for now
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


def main():
    parser = argparse.ArgumentParser(description="LLM Inference Benchmark")
    parser.add_argument("--prompt", type=str, default="Hello, world", help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Number of tokens to generate")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (if supported in engine)")

    args = parser.parse_args()

    config = ServingConfig(device=args.device)
    config.compile_model = args.compile

    engine = LLMEngine(config)
    engine.load_model()

    try:
        run_benchmark(engine, args.prompt, args.max_new_tokens, args.runs)
    finally:
        engine.unload_model()


if __name__ == "__main__":
    main()
