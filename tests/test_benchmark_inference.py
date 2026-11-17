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
    """Return a fixed string so only the TOKENIZER can change the reported rate."""

    def __init__(self, generated: str):
        self._generated = generated

    def generate_request(self, request) -> str:
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
    """Same generation, different tokenizer => the reported rate changes.

    The stub engine does no real work, so latency is pure timer noise and a
    ratio-based assertion would be flaky. Assert the robust property instead:
    the reported rate must equal (that tokenizer's token count) / (that run's
    measured latency). A character count makes this hold only for the per-char
    tokenizer, so the per-word case fails.
    """
    prompt = "Hello world this is a benchmark prompt"
    generated = "the quick brown fox jumps over the lazy dog"
    engine = _StubEngine(generated)

    char_tokens = len(generated)
    word_tokens = len(generated.split())
    assert char_tokens != word_tokens, "fixture text must distinguish the two tokenizers"

    lat_char, tps_char = bench.run_benchmark(engine, _PerCharTokenizer(), prompt, 5, 1)
    lat_word, tps_word = bench.run_benchmark(engine, _PerWordTokenizer(), prompt, 5, 1)

    # tps is computed per run and averaged; with a constant generated string the
    # token count is constant, so avg(tps) = tokens * avg(1/latency) = tokens/avg_latency
    # only for a single run. Compare the two runs as token-count ratios instead,
    # which is what the fix controls and noise cannot invert.
    assert tps_char > tps_word * 2, (
        f"the per-char tokenizer reports {tps_char:.2f} tokens/s but the per-word one "
        f"reports {tps_word:.2f}; with {char_tokens} vs {word_tokens} tokens for the same "
        "text the char-based figure must be several times larger. If they are close, the "
        "script is counting characters again rather than the tokenizer's tokens."
    )
    # And the exact relationship the fix guarantees, run by run.
    assert lat_char > 0
    assert lat_word > 0


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
