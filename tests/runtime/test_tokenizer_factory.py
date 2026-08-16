"""Tests for TokenizerFactory."""

import pytest
import torch

from llm.runtime.tokenizer_factory import TokenizerFactory
from llm.serving.config import ServingConfig
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer
from llm.training.core.config import Config


def test_from_data_config_simple_fallback():
    config = Config()
    tokenizer = TokenizerFactory.from_data_config(config.data)
    assert tokenizer.vocab_size > 3
    assert tokenizer.decode(tokenizer.encode("<PAD>")) == "<PAD>"


def test_from_data_config_pickle(tmp_path):
    config = Config()
    tokenizer = SimpleCharacterTokenizer(["abc"])
    path = tmp_path / "tok.pt"
    torch.save(tokenizer, path)
    config.data.tokenizer_path = str(path)

    loaded = TokenizerFactory.from_data_config(config.data)
    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.encode("abc") == tokenizer.encode("abc")


def test_from_data_config_missing_simple_path_raises(tmp_path):
    """A configured-but-missing ``simple`` tokenizer must fail loud.

    Before this fix ``from_data_config`` silently fell back to the default
    corpus tokenizer when ``tokenizer_path`` pointed at a nonexistent file,
    so training quietly proceeded with a different vocabulary than the one
    used to encode the data — the checkpoint could never round-trip with the
    intended tokenizer. This mirrors ``from_serving_config`` (which raises
    FileNotFoundError for the same condition).
    """
    config = Config()
    config.data.tokenizer_type = "simple"
    config.data.tokenizer_path = str(tmp_path / "does_not_exist.pt")

    with pytest.raises(FileNotFoundError, match="Tokenizer file not found"):
        TokenizerFactory.from_data_config(config.data)


def test_from_data_config_no_path_uses_default_corpus():
    """No ``tokenizer_path`` at all is the documented default path: build a
    simple tokenizer from the default corpus (existing behavior preserved)."""
    config = Config()
    config.data.tokenizer_type = "simple"
    config.data.tokenizer_path = None

    tokenizer = TokenizerFactory.from_data_config(config.data)
    assert isinstance(tokenizer, SimpleCharacterTokenizer)
    assert tokenizer.vocab_size > 3


def test_from_data_config_default_tokenizer_encodes_real_text():
    """Regression (RIL ISS-196): the documented default simple tokenizer
    (``tokenizer_type="simple"`` with no path) must be able to encode real
    text. The old default corpus ``["<PAD>", "<EOS>", "<BOS>"]`` produced a
    vocab of only the three markers' constituent characters, so any real
    corpus raised ``KeyError`` on the first non-marker character — the
    default config was unusable for actual training data.
    """
    config = Config()
    config.data.tokenizer_type = "simple"
    config.data.tokenizer_path = None

    tokenizer = TokenizerFactory.from_data_config(config.data)
    sample = "The quick brown fox jumps over the lazy dog, 123!"
    encoded = tokenizer.encode(sample)
    assert tokenizer.decode(encoded) == sample
    assert tokenizer.pad_token_id is not None


def test_from_data_config_hf_requires_path():
    config = Config()
    config.data.tokenizer_type = "hf"
    config.data.tokenizer_path = None

    with pytest.raises(ValueError, match="tokenizer_path"):
        TokenizerFactory.from_data_config(config.data)


def test_from_serving_config_requires_tokenizer_with_model(tmp_path):
    ckpt = tmp_path / "model.pt"
    ckpt.write_text("x", encoding="utf-8")
    config = ServingConfig(model_path=str(ckpt))

    with pytest.raises(ValueError, match="tokenizer_path is required"):
        TokenizerFactory.from_serving_config(config)


def test_from_serving_config_default_simple():
    config = ServingConfig()
    tokenizer = TokenizerFactory.from_serving_config(config)
    roundtrip = tokenizer.decode(tokenizer.encode("Hello"))
    assert roundtrip == "Hello"


def test_from_dataset_text_round_trips_file_contents(tmp_path):
    """Requirement: from_dataset_text builds a vocab covering every character in the file."""
    data_file = tmp_path / "corpus.txt"
    data_file.write_text("cab", encoding="utf-8")

    tokenizer = TokenizerFactory.from_dataset_text(data_file)

    assert tokenizer.decode(tokenizer.encode("cab")) == "cab"
    assert len(tokenizer.encode("cab")) == 3


def test_from_dataset_text_registers_eos_bos_specials(tmp_path):
    """Regression (RIL ISS-152): ``from_dataset_text`` builds its corpus as
    ``["<PAD>", "<EOS>", "<BOS>", *chars]`` intending all three markers to be
    registered as special tokens. Only ``<PAD>`` was special-cased — the
    ``<EOS>``/``<BOS>`` markers were flattened into their constituent plain
    characters and ``eos_token_id``/``bos_token_id`` stayed ``None``, so
    eval generations (LMTask / lm_eval generate_until) never stopped on the
    model's EOS. The eval tokenizer must expose real EOS/BOS token ids."""
    data_file = tmp_path / "corpus.txt"
    data_file.write_text("cab", encoding="utf-8")

    tokenizer = TokenizerFactory.from_dataset_text(data_file)

    assert tokenizer.eos_token_id is not None
    assert tokenizer.bos_token_id is not None
    # The special markers must encode to exactly one token (not the char ids
    # of '<', 'E', 'O', 'S', '>').
    assert tokenizer.encode(SimpleCharacterTokenizer.eos_char) == [tokenizer.eos_token_id]
    assert tokenizer.encode(SimpleCharacterTokenizer.bos_char) == [tokenizer.bos_token_id]
    assert tokenizer.decode([tokenizer.eos_token_id]) == SimpleCharacterTokenizer.eos_char
    assert tokenizer.decode([tokenizer.bos_token_id]) == SimpleCharacterTokenizer.bos_char


def test_tokenizer_loader_refuses_malicious_pickle(tmp_path):
    """Regression (RIL ISS-185): ``TokenizerFactory`` on an attacker-controlled
    ``tokenizer.pt`` must NOT run arbitrary ``__reduce__`` code. The old
    ``weights_only=False`` load executed the payload (os.system-touch a marker);
    the hardened loader refuses via the allowlist and no marker is created."""
    import os
    import pickle

    marker = tmp_path / "pwned"
    payload = tmp_path / "evil.pt"

    class _Exploit:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))

    with payload.open("wb") as f:
        pickle.dump(_Exploit(), f)

    class _Cfg:
        tokenizer_type = "simple"
        tokenizer_path = str(payload)

    from llm.runtime.tokenizer_factory import TokenizerFactory

    # The hardened loader refuses the foreign global (os.system) — the exact
    # class torch's weights_only unpickler raises.
    with pytest.raises(Exception, match=r"global|weights_only|load"):
        TokenizerFactory.from_serving_config(_Cfg())

    assert not marker.exists(), "malicious pickle executed code during tokenizer load"


def test_tokenizer_roundtrip_still_works_weights_only(tmp_path):
    """The hardening must not break the legit framework round-trip: a pickle
    written by ``torch.save(SimpleCharacterTokenizer(...))`` loads fine."""
    from llm.runtime.tokenizer_factory import TokenizerFactory

    config = Config()
    tokenizer = SimpleCharacterTokenizer(["abc", "<PAD>"])
    path = tmp_path / "tok.pt"
    torch.save(tokenizer, path)
    config.data.tokenizer_path = str(path)

    loaded = TokenizerFactory.from_data_config(config.data)
    assert loaded.encode("<PAD>") == tokenizer.encode("<PAD>")
