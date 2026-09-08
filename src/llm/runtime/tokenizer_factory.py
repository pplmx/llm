"""Central tokenizer loading for training, serving, and evaluation."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Protocol, cast

import torch

from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer
from llm.tokenization.tokenizer import BaseTokenizer, HFTokenizer


def _default_simple_corpus() -> list[str]:
    """The documented default corpus for a ``simple`` tokenizer with no path.

    ``string.printable`` so real text is encodable (RIL ISS-196: the old
    corpus of just the three markers produced a vocab of only their
    constituent characters, so the default config raised ``KeyError`` on any
    real corpus), plus the three markers so ``<PAD>``/``<EOS>``/``<BOS>`` are
    still registered as multi-char special tokens.
    """
    import string

    return [string.printable, "<PAD>", "<EOS>", "<BOS>"]


DEFAULT_SIMPLE_CORPUS = _default_simple_corpus()

_SAFE_GLOBALS_REGISTERED = False


def _register_tokenizer_safe_globals() -> None:
    """Allowlist the framework's tokenizer classes and the transformers
    tokenizer base, so ``torch.load(..., weights_only=True)`` can reconstruct
    a user-saved ``tokenizer.pt`` WITHOUT executing arbitrary pickled code.

    Security (RIL ISS-185): before this, both ``from_data_config`` and
    ``from_serving_config`` loaded a user-supplied ``tokenizer.pt`` with
    ``weights_only=False`` — an unrestricted ``pickle.load`` where a crafted
    ``__reduce__`` (e.g. ``os.system``) ran as the training/serving process
    before any input reached the model. With the allowlist, a pickle that
    references any class OUTSIDE these (an attacker's ``os.system`` /
    ``subprocess`` / a genuinely custom undeclared tokenizer class) raises
    UnpicklingError and is refused — no ``weights_only=False`` fallback
    (consistent with the round-62 checkpoint hardening, ISS-170).
    """
    global _SAFE_GLOBALS_REGISTERED
    if _SAFE_GLOBALS_REGISTERED:
        return
    classes: list[Any] = [BaseTokenizer, SimpleCharacterTokenizer, HFTokenizer]
    # An HFTokenizer pickle embeds the wrapped transformers object; allowlist
    # the tokenizer base classes when the optional dependency is present
    # (transformers absent is fine — the allowlist stays framework-only).
    import contextlib

    with contextlib.suppress(Exception):
        from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast

        classes.extend([PreTrainedTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast])
    # ty models add_safe_globals' arg as callables; classes are callables at
    # runtime but ty widens them to ``type`` — ignore the arg-type nit.
    torch.serialization.add_safe_globals(classes)  # type: ignore[arg-type]
    _SAFE_GLOBALS_REGISTERED = True


def _load_tokenizer_pt(path: Path) -> Any:
    """``torch.load`` a tokenizer pickle under ``weights_only=True`` with an
    actionable failure message.

    Any pickle whose embedded globals fall outside the safe-globals allowlist
    — a genuinely malicious payload, OR a real HF ``PreTrainedTokenizerFast``
    whose pickle state carries ``tokenizers``/``sentencepiece`` rust objects
    the allowlist cannot enumerate (RIL ISS-413) — raises ``UnpicklingError``
    from torch's weights-only unpickler. Both cases must fail loud with a
    remediation hint; the HF case is user error (an HF tokenizer should be
    configured as ``tokenizer_type='hf'`` with a repo id/directory, not saved
    to a ``.pt``).
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.PickleError, RuntimeError, EOFError, AttributeError) as exc:
        raise pickle.UnpicklingError(
            f"cannot load tokenizer pickle {path} under weights_only=True: {exc}. "
            "If this is a HuggingFace tokenizer, use tokenizer_type='hf' with the "
            "repo id (or local directory) instead of a .pt file."
        ) from exc


def _ensure_tokenizer_obj(loaded: Any, path: str) -> BaseTokenizer:
    """Type-check a loaded tokenizer pickle, failing loud on a wrong file.

    A ``tokenizer_path`` that actually points at a model blob (state dict /
    full-checkpoint dict) loads cleanly under ``weights_only=True`` but is not
    a tokenizer — the mistake only surfaces later as a confusing
    ``AttributeError: 'dict' object has no attribute 'encode'`` at the first
    request. Catch it here (RIL ISS-420).
    """
    # ``BaseTokenizer`` is a plain (non runtime-checkable) Protocol, so a
    # structural check on the contract methods is the safe test here.
    if callable(getattr(loaded, "encode", None)) and callable(getattr(loaded, "decode", None)):
        return cast(BaseTokenizer, loaded)
    raise ValueError(
        f"Tokenizer file {path!r} loaded as {type(loaded).__name__}, not a tokenizer. "
        "Point tokenizer_path at a tokenizer .pt (produced by "
        "torch.save(SimpleCharacterTokenizer(...))) or an HF directory."
    )


class TokenizerConfig(Protocol):
    tokenizer_type: str
    tokenizer_path: str | None


class TokenizerFactory:
    """Load tokenizers from training DataConfig or ServingConfig duck-typed objects."""

    @staticmethod
    def from_data_config(
        data_config: TokenizerConfig,
        *,
        default_corpus: list[str] | None = None,
    ) -> BaseTokenizer:
        if data_config.tokenizer_type == "hf":
            if not data_config.tokenizer_path:
                raise ValueError("tokenizer_path must be specified for HF tokenizer.")
            return cast(BaseTokenizer, HFTokenizer.from_pretrained(data_config.tokenizer_path))

        if data_config.tokenizer_path:
            path = Path(data_config.tokenizer_path)
            if not path.exists():
                # A configured-but-missing ``simple`` tokenizer must fail
                # loud, exactly like ``from_serving_config`` (raise
                # FileNotFoundError).  Silently substituting the default
                # corpus tokenizer here means training proceeds with a
                # different vocabulary than the one used to build the data /
                # checkpoint — every epoch trains (and saves) against a
                # tokenizer that can't round-trip the intended vocab, and the
                # mismatch is invisible until serve-time decode errors.
                raise FileNotFoundError(f"Tokenizer file not found: {path}")
            _register_tokenizer_safe_globals()
            return _ensure_tokenizer_obj(_load_tokenizer_pt(path), str(path))

        if default_corpus is None:
            # ``or``-fallback would silently REPLACE an explicitly-passed
            # empty corpus with the printable default — honou the parameter,
            # only ``None`` selects the default (RIL ISS-420).
            default_corpus = DEFAULT_SIMPLE_CORPUS
        return cast(BaseTokenizer, SimpleCharacterTokenizer(default_corpus))

    @staticmethod
    def from_serving_config(config: Any) -> Any:
        if config.tokenizer_path:
            path = Path(config.tokenizer_path)
            if config.tokenizer_type == "hf":
                return HFTokenizer.from_pretrained(config.tokenizer_path)
            if not path.exists():
                raise FileNotFoundError(f"Tokenizer file not found: {path}")
            _register_tokenizer_safe_globals()
            return _ensure_tokenizer_obj(_load_tokenizer_pt(path), str(path))

        # ``tokenizer_type="hf"`` with no path must fail loud, exactly like
        # ``from_data_config`` — falling through to the printable-corpus char
        # tokenizer here would silently serve with a vocab that cannot encode
        # real text, surfacing as decode garbage only after startup.
        if getattr(config, "tokenizer_type", None) == "hf":
            raise ValueError("tokenizer_path must be specified for HF tokenizer.")

        if getattr(config, "model_path", None):
            raise ValueError("tokenizer_path is required when model_path is set for serving")

        return TokenizerFactory.from_printable_corpus()

    @staticmethod
    def from_printable_corpus() -> SimpleCharacterTokenizer:
        # Printable corpus PLUS the PAD/EOS/BOS markers (RIL TASK-327): a
        # plain printable-only tokenizer has ``eos_token_id None``, and every
        # generation backend silently treats a ``None`` EOS as "never stop" —
        # serving's dummy-model fallback (the only production consumer) ran
        # every request to ``max_new_tokens``. ``DEFAULT_SIMPLE_CORPUS`` is the
        # same printable set plus the three declared specials, so this matches
        # the training default (ISS-152 already fixed the train/eval side).
        return SimpleCharacterTokenizer(DEFAULT_SIMPLE_CORPUS)

    @staticmethod
    def from_default_test_corpus() -> SimpleCharacterTokenizer:
        corpus = [
            "hello world",
            "the quick brown fox",
            "testing one two three",
            "abcdefghijklmnopqrstuvwxyz",
        ]
        return SimpleCharacterTokenizer(corpus)

    @staticmethod
    def from_dataset_text(dataset_path: str | Path) -> SimpleCharacterTokenizer:
        """Build a character tokenizer from the unique characters in a text file."""
        try:
            text = Path(dataset_path).read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{dataset_path} is not valid UTF-8 ({exc}); from_dataset_text builds "
                "a character tokenizer from decoded text — decode the file with your "
                "own step first (RIL ISS-420)."
            ) from exc
        chars = sorted(set(text))
        corpus = ["<PAD>", "<EOS>", "<BOS>", *chars]
        return SimpleCharacterTokenizer(corpus)

    @staticmethod
    def cache_hf_tokenizer(data_config: TokenizerConfig) -> None:
        if data_config.tokenizer_type == "hf" and data_config.tokenizer_path:
            HFTokenizer.from_pretrained(data_config.tokenizer_path)
