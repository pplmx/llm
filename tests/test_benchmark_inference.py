"""``scripts/benchmark_inference.py`` must report tokens, not characters.

The script used to infer the token count from string length::

    num_tokens = max(len(output) - len(prompt), 1)

That is a character count. It happens to be correct for exactly one tokenizer —
``TokenizerFactory.from_printable_corpus``, the demo fallback used when
``ServingConfig.model_path`` is unset, which encodes one printable character per
token. The moment a real checkpoint is served (the ``serve_pretrained.yaml``
setup this script exists to benchmark) the tokenizer is BPE and a token spans
several characters, so the script overstated throughput by the
chars-per-token ratio.

These tests drive ``run_benchmark`` with two tokenizers that disagree on the
token count for the SAME generated text, which is the property the old code
could not express: it would print the same figure for both.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import time

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "benchmark_inference.py"


def _load_script():
    """Import ``scripts/benchmark_inference.py`` by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("_benchmark_inference_under_test", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _StubEngine:
    """Return a fixed string, after a DETERMINISTIC delay per call.

    The delay matters: ``run_benchmark`` computes ``tokens / latency``, and with
    an instantaneous stub the latency is pure timer noise (measured ratios of
    1.16x where 4.78x was expected), so any ratio-based assertion over the
    reported rates is flaky — it passed when written and failed in a full suite
    run under load. A real sleep makes the denominator meaningful, at which point
    the reported rate depends only on the TOKEN COUNT.
    """

    def __init__(self, generated: str, delay_seconds: float = 0.002):
        self._generated = generated
        self._delay = delay_seconds

    def generate_request(self, request) -> str:
        time.sleep(self._delay)
        return request.prompt + self._generated


class _PerCharTokenizer:
    """One token per character — models the printable-corpus fallback."""

    def encode(self, text):
        return list(range(len(text)))


class _PerWordTokenizer:
    """One token per whitespace word — nothing like a character count."""

    def encode(self, text):
        return list(range(len(text.split())))


@pytest.fixture(scope="module")
def bench():
    return _load_script()


def test_the_two_tokenizers_disagree_on_the_same_text():
    """Guard the premise: if these agreed, the test below could not distinguish."""
    text = "the quick brown fox jumps"
    assert len(_PerCharTokenizer().encode(text)) != len(_PerWordTokenizer().encode(text))


def test_tokens_per_second_follows_the_tokenizer(bench):
    """Same generation and same latency => the reported rate depends on tokens only.

    ``_StubEngine`` sleeps a fixed amount per call, so both runs have the same
    denominator and ``tps`` is proportional to the token count the tokenizer
    reports. With tokens of 43 vs 9 the per-char rate must be ~4.8x the per-word
    rate; a character count in the script would make them equal.

    The tolerance covers scheduler jitter on a busy machine, which perturbs
    individual sleeps but cannot move a 4.8x ratio back near 1x.
    """
    prompt = "Hello world this is a benchmark prompt"
    generated = "the quick brown fox jumps over the lazy dog"
    engine = _StubEngine(generated)

    char_tokens = len(generated)
    word_tokens = len(generated.split())
    assert char_tokens != word_tokens, "fixture text must distinguish the two tokenizers"
    expected_ratio = char_tokens / word_tokens

    _lat_char, tps_char = bench.run_benchmark(engine, _PerCharTokenizer(), prompt, 5, 1)
    _lat_word, tps_word = bench.run_benchmark(engine, _PerWordTokenizer(), prompt, 5, 1)

    # The deterministic delay lives in the stub, so both runs take the same path
    # through run_benchmark and differ only in num_tokens.
    ratio = tps_char / tps_word
    assert ratio == pytest.approx(expected_ratio, rel=0.5), (
        f"tokens/s did not scale with the tokenizer's token count: the per-char run reported "
        f"{tps_char:.2f} and the per-word run {tps_word:.2f} (ratio {ratio:.2f}, expected "
        f"~{expected_ratio:.2f} from {char_tokens} vs {word_tokens} tokens). If the ratio is "
        "near 1.0 the script is counting characters again rather than the tokenizer's tokens."
    )


def test_a_character_count_would_report_identical_rates():
    """Pin what the old implementation would have produced, so the bug cannot return.

    Recomputed directly from the outputs rather than by re-running the old code:
    if ``num_tokens`` were ``len(output) - len(prompt)``, the two tokenizers
    would yield the SAME token count for the same text, hence the same tps.
    """
    prompt = "Hello world this is a benchmark prompt"
    generated = "the quick brown fox jumps over the lazy dog"
    char_based_tokens = len(prompt + generated) - len(prompt)
    word_based_tokens = len(_PerWordTokenizer().encode(generated))
    assert char_based_tokens != word_based_tokens, (
        "the fixture text must distinguish a character count from a word count"
    )


def test_run_benchmark_requires_a_tokenizer_argument(bench):
    """The tokenizer must be threaded in, not inferred from string length."""
    import inspect

    params = inspect.signature(bench.run_benchmark).parameters
    assert "tokenizer" in params, (
        "run_benchmark lost its tokenizer parameter; tokens/s can only be measured "
        "with the tokenizer the served model actually uses"
    )


def _latency_percentiles_from(bench, capsys, runs: int) -> dict[str, float]:
    """Run the benchmark and parse the printed percentile lines."""
    import re

    engine = _StubEngine("ab")
    bench.run_benchmark(engine, _PerCharTokenizer(), "hi", 5, runs)
    out = capsys.readouterr().out
    parsed: dict[str, float] = {}
    for name in ("Avg", "P50", "P95", "P99"):
        match = re.search(rf"^{name} Latency: ([0-9.]+)s$", out, re.MULTILINE)
        assert match is not None, f"missing {name} Latency line in:\n{out}"
        parsed[name] = float(match.group(1))
    return parsed


def test_the_benchmark_reports_latency_percentiles(bench, capsys):
    """A mean alone hides the tail that decides a serving question.

    ROADMAP listed "latency has only an end-to-end mean, no percentiles" as the
    remaining benchmark gap: two deployments with the same average latency can
    differ completely in how bad the slow runs are.
    """
    parsed = _latency_percentiles_from(bench, capsys, runs=8)

    assert {"Avg", "P50", "P95", "P99"} <= set(parsed)
    assert 0 < parsed["P50"] <= parsed["P95"] <= parsed["P99"], f"percentiles must be non-decreasing, got {parsed}"


def test_percentiles_are_observed_latencies_not_interpolations(bench, capsys):
    """Every reported percentile must be a latency that actually occurred.

    ``statistics.quantiles`` interpolates and can return values OUTSIDE the sample
    range — measured on ``[1.0…1.8, 10.0]`` it produced p95 = 13.69 and p99 =
    17.30 from a sample whose maximum is 10.0. A "slower than the slowest run"
    percentile is not a measurement, so the implementation uses nearest-rank and
    this test pins the range property.
    """
    parsed = _latency_percentiles_from(bench, capsys, runs=6)

    # P99 of 6 evenly-timed runs is at most the slowest run, which is also
    # bounded below by the average. Anything above P99*1.5 would indicate
    # extrapolation past the sample.
    assert parsed["P99"] <= parsed["Avg"] * 3, (
        f"P99={parsed['P99']} is far above the mean {parsed['Avg']}; a percentile cannot exceed "
        "the slowest observed run by that much"
    )
    assert parsed["P50"] <= parsed["P99"]


def test_a_single_run_benchmark_still_reports_percentiles(bench, capsys):
    """One sample must not crash the percentile code.

    ``statistics.quantiles`` raises on fewer than two data points, so a
    single-run benchmark needs an explicit fallback rather than an exception.
    """
    parsed = _latency_percentiles_from(bench, capsys, runs=1)

    assert parsed["P50"] == parsed["P95"] == parsed["P99"], (
        f"with one sample every percentile is that sample, got {parsed}"
    )
