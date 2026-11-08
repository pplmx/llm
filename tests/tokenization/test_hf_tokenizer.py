"""Tests for :class:`llm.tokenization.tokenizer.HFTokenizer`.

The class had no tests at all — 13 of its 40 statements were uncovered, the
whole of it reachable with a locally-built tokenizer directory (no network).
The gap mattered because of the ``pad_token`` fallback in ``__init__``: an HF
tokenizer that ships without a pad token gets ``pad = eos``, and
``serving/batch_engine.py`` has a dedicated branch for that case (masking the
pad id would otherwise suppress the EOS halt signal and every request would run
to ``max_new_tokens``). Nothing pinned the fallback, so the shared-id
configuration the engine special-cases rested on an untested assumption.

The fixtures build a tiny ``WordLevel`` tokenizer and save it to ``tmp_path``,
so these tests exercise the real ``AutoTokenizer.from_pretrained`` path without
touching the network or the HF cache.
"""

# ruff: noqa: S106
# S106 (hardcoded password) fires on the tokenizer keyword arguments below
# (``unk_token`` / ``eos_token`` / ``pad_token`` / ``bos_token``). They are
# vocabulary entries for a throwaway test tokenizer, not credentials.

from __future__ import annotations

import pathlib

import pytest

from llm.tokenization.tokenizer import HFTokenizer

pytest.importorskip("transformers", reason="transformers is an optional dependency")


def _write_tokenizer(
    directory: pathlib.Path,
    *,
    vocab: dict[str, int],
    with_pad: bool,
) -> pathlib.Path:
    """Save a minimal ``WordLevel`` tokenizer to ``directory`` and return it."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<eos>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<eos>",
        pad_token="<pad>" if with_pad else None,
    )
    fast.save_pretrained(str(directory))
    return directory


@pytest.fixture
def tokenizer_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """A tokenizer that DOES declare a pad token."""
    return _write_tokenizer(
        tmp_path / "with_pad",
        vocab={"<pad>": 0, "<eos>": 1, "hello": 2, "world": 3},
        with_pad=True,
    )


@pytest.fixture
def tokenizer_dir_no_pad(tmp_path: pathlib.Path) -> pathlib.Path:
    """A tokenizer with an EOS but no pad — the GPT-2-style configuration."""
    return _write_tokenizer(
        tmp_path / "no_pad",
        vocab={"<eos>": 0, "hello": 1, "world": 2},
        with_pad=False,
    )


@pytest.fixture
def tokenizer_dir_with_template(tmp_path: pathlib.Path) -> pathlib.Path:
    """A tokenizer whose post-processor WRAPS the sequence in special tokens.

    The plain fixtures above encode ``"hello world"`` to the same ids whether or
    not special tokens are requested, which makes an
    ``add_special_tokens=False`` assertion vacuous — removing the flag left the
    test green. A template processor (BOS … EOS) makes the two calls differ, so
    the assertion can actually fail.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    directory = tmp_path / "with_template"
    backend = Tokenizer(
        models.WordLevel(
            vocab={"<pad>": 0, "<eos>": 1, "<bos>": 2, "hello": 3, "world": 4},
            unk_token="<eos>",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[("<bos>", 2), ("<eos>", 1)],
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<eos>",
        pad_token="<pad>",
        bos_token="<bos>",
    )
    fast.save_pretrained(str(directory))
    return directory


class TestConstructionAndProperties:
    def test_loads_from_a_local_directory(self, tokenizer_dir):
        tokenizer = HFTokenizer.from_pretrained(str(tokenizer_dir))
        assert tokenizer.vocab_size == 4

    def test_special_token_ids_are_exposed(self, tokenizer_dir):
        tokenizer = HFTokenizer.from_pretrained(str(tokenizer_dir))
        assert tokenizer.pad_token_id == 0
        assert tokenizer.eos_token_id == 1

    def test_bos_token_id_is_none_when_absent(self, tokenizer_dir):
        """A missing BOS must stay ``None`` rather than being invented."""
        tokenizer = HFTokenizer.from_pretrained(str(tokenizer_dir))
        assert tokenizer.bos_token_id is None

    def test_constructor_accepts_a_directory_string(self, tokenizer_dir):
        """The public constructor path (``from_pretrained`` delegates to it)."""
        tokenizer = HFTokenizer(str(tokenizer_dir))
        assert tokenizer.vocab_size == 4


class TestPadTokenFallback:
    def test_missing_pad_falls_back_to_eos(self, tokenizer_dir_no_pad):
        """A tokenizer with no pad gets ``pad = eos``.

        This is the configuration ``batch_engine`` special-cases: when pad and
        eos are the SAME id, masking the pad logit would also mask the halt
        signal, so the engine skips the mask instead. The fallback living here
        is what makes that shared id exist, so it is pinned.
        """
        tokenizer = HFTokenizer.from_pretrained(str(tokenizer_dir_no_pad))
        assert tokenizer.eos_token_id == 0
        assert tokenizer.pad_token_id == tokenizer.eos_token_id, (
            "a tokenizer without a pad token must adopt its eos token as pad"
        )

    def test_explicit_pad_token_is_not_overwritten(self, tokenizer_dir):
        """The fallback must not clobber a tokenizer that already has a pad."""
        tokenizer = HFTokenizer.from_pretrained(str(tokenizer_dir))
        assert tokenizer.pad_token_id == 0
        assert tokenizer.eos_token_id == 1
        assert tokenizer.pad_token_id != tokenizer.eos_token_id


class TestEncodeDecode:
    def test_encode_produces_ids_without_special_tokens(self, tokenizer_dir_with_template):
        """``add_special_tokens=False`` keeps BOS/EOS out of the sequence.

        The training and serving paths compute lengths from this output; an
        implicitly added EOS would shift every position id by one.

        Uses the *template* tokenizer on purpose. With a plain tokenizer the
        flag makes no difference and the assertion would pass either way — this
        was verified by deleting the flag, which left the test green until the
        fixture was switched. Here the template would otherwise inject
        ``<bos> … <eos>``.
        """
        tokenizer = HFTokenizer.from_pretrained(str(tokenizer_dir_with_template))
        assert tokenizer.encode("hello world") == [3, 4], (
            "special tokens leaked into the encoded ids: the tokenizer's template "
            "wraps the sequence, so this should be the bare payload"
        )

    def test_encode_roundtrips_through_decode(self, tokenizer_dir):
        tokenizer = HFTokenizer.from_pretrained(str(tokenizer_dir))
        text = "hello world"
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_decode_returns_a_real_string(self, tokenizer_dir):
        """``decode`` asserts its own return type; that assertion must hold."""
        tokenizer = HFTokenizer.from_pretrained(str(tokenizer_dir))
        decoded = tokenizer.decode([2, 3])
        assert isinstance(decoded, str)
        assert decoded


class TestSavePretrained:
    def test_saved_directory_can_be_reloaded(self, tokenizer_dir, tmp_path):
        """A saved copy must round-trip through ``from_pretrained``.

        Checkpoints save the tokenizer beside the weights; if that copy cannot
        be reloaded, the resumed run cannot tokenize its own data.
        """
        tokenizer = HFTokenizer.from_pretrained(str(tokenizer_dir))
        out = tmp_path / "resaved"
        tokenizer.save_pretrained(str(out))

        assert (out / "tokenizer_config.json").exists()
        reloaded = HFTokenizer.from_pretrained(str(out))
        assert reloaded.vocab_size == tokenizer.vocab_size
        assert reloaded.encode("hello world") == tokenizer.encode("hello world")
