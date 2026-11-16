"""HF tokenizer integration smoke test — runs offline.

This module used to call ``HFTokenizer.from_pretrained("gpt2")`` against the
Hub and skip on any transport failure. That made it untested everywhere the
network is unavailable or proxied (this host included: an ambient ``no_proxy``
entry of ``[::1]`` makes httpx raise ``InvalidURL: Invalid port: ':1]'`` before a
socket opens), and its assertions pinned gpt2's own token ids — facts about an
upstream model, not about this repo.

``tests/tokenization/test_hf_tokenizer.py`` already establishes the offline
pattern: build a tiny ``WordLevel`` tokenizer, save it to ``tmp_path``, and let
``AutoTokenizer.from_pretrained`` read it from disk. The same contract is
checked here — a directory loads through the public API and round-trips text —
so the test is no longer hostage to egress.
"""

# ruff: noqa: S106
# S106 (hardcoded password) fires on the tokenizer keyword arguments below
# (``unk_token`` / ``eos_token``). They are vocabulary entries for a throwaway
# test tokenizer, not credentials — the same exemption the sibling
# ``tests/tokenization/test_hf_tokenizer.py`` carries.

from __future__ import annotations

import pathlib

import pytest

from llm.tokenization.tokenizer import HFTokenizer

pytest.importorskip("transformers", reason="transformers is an optional dependency")


def _write_tokenizer(directory: pathlib.Path) -> pathlib.Path:
    """Save a minimal ``WordLevel`` tokenizer to ``directory``."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel(vocab={"<eos>": 0, "hello": 1, "world": 2}, unk_token="<eos>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>").save_pretrained(str(directory))
    return directory


def test_hf_tokenizer_round_trips_a_saved_directory(tmp_path: pathlib.Path):
    """``HFTokenizer`` loads a tokenizer DIRECTORY and round-trips text.

    The assertion is on the round trip, which is the property callers rely on —
    not on any particular model's vocabulary ids.
    """
    tokenizer = HFTokenizer.from_pretrained(str(_write_tokenizer(tmp_path / "tok")))

    text = "hello world"
    tokens = tokenizer.encode(text)
    assert len(tokens) == 2, f"expected one token per word, got {tokens}"
    assert tokenizer.decode(tokens) == text

    # The vocabulary is the one on disk, so an unknown word falls to the
    # configured unk token instead of raising.
    assert tokenizer.encode("notinvocab")
