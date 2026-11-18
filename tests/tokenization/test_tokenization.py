import pytest

from llm.tokenization.bpe_tokenizer import BPETokenizer


@pytest.fixture
def sample_text_file(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    f = d / "sample.txt"
    f.write_text("hello world\nhello universe\nworld of wonders", encoding="utf-8")
    return str(f)


@pytest.mark.quick
def test_bpe_tokenizer_train_save_load(sample_text_file, tmp_path):
    # Train
    tokenizer = BPETokenizer.train([sample_text_file], vocab_size=100, min_frequency=1)

    text = "hello world"
    encoded = tokenizer.encode(text)
    assert tokenizer.vocab_size > len(encoded)
    assert tokenizer.pad_token_id >= 0
    assert tokenizer.decode(encoded) == text

    # Save
    save_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(save_path))
    assert save_path.exists()

    # Load
    loaded_tokenizer = BPETokenizer.load(str(save_path))
    assert loaded_tokenizer.vocab_size == tokenizer.vocab_size

    # Encode / Decode with loaded
    text2 = "universe"
    assert loaded_tokenizer.decode(loaded_tokenizer.encode(text2)) == text2


@pytest.mark.quick
def test_bpe_tokenizer_special_tokens(sample_text_file):
    special_tokens = ["[UNK]", "[PAD]", "[MASK]", "CustomToken"]
    tokenizer = BPETokenizer.train([sample_text_file], vocab_size=100, min_frequency=1, special_tokens=special_tokens)

    vocab = tokenizer.get_vocab()
    for token in special_tokens:
        assert token in vocab


@pytest.mark.quick
def test_bpe_tokenizer_empty_input():
    tokenizer = BPETokenizer()  # Default initialized un-trained
    # Note: Default initialized might not encode well without training,
    # but the method should not crash.
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""


@pytest.mark.quick
def test_bpe_pad_token_id_is_valid_when_pad_missing_from_vocab(sample_text_file):
    """Regression: ``pad_token_id`` returned
    ``token_to_id("[PAD]")`` which is ``None`` whenever the trained/loaded
    vocab does not contain ``[PAD]`` (custom ``special_tokens``, or a
    foreign ``tokenizer.json``). Callers use ``getattr(tokenizer,
    'pad_token_id', 0)`` — the attribute *exists* as None so the fallback
    never fires, and padding builds ``[None] * n`` which crashes
    ``torch.tensor`` in ``batch_generate`` / ``TextDataset.__getitem__``.

    The documented convention (text.py) is "no pad id → 0"; the property
    must never return ``None``."""
    special_tokens = ["[UNK]", "[MASK]", "CustomToken"]  # deliberately no [PAD]
    tokenizer = BPETokenizer.train([sample_text_file], vocab_size=100, min_frequency=1, special_tokens=special_tokens)

    assert tokenizer.pad_token_id is not None
    assert tokenizer.pad_token_id >= 0
    # The safe fallback must not collide with a real token's id.
    assert 0 <= tokenizer.pad_token_id < tokenizer.vocab_size


@pytest.mark.quick
def test_bpe_tokenizer_save_creates_parent_dirs(sample_text_file, tmp_path):
    """``save`` should create the parent directory if it does not exist."""
    tokenizer = BPETokenizer.train([sample_text_file], vocab_size=100, min_frequency=1)

    nested_save_path = tmp_path / "nested" / "subdir" / "tokenizer.json"
    tokenizer.save(str(nested_save_path))
    assert nested_save_path.exists()

    loaded = BPETokenizer.load(str(nested_save_path))
    assert loaded.vocab_size == tokenizer.vocab_size


# --- the lowercase normalizer: the one deliberately LOSSY knob ------------
#
# ``lowercase`` defaults to True, and the class docstring states plainly that
# this makes the tokenizer lossy -- ``decode(encode(x))`` is NOT ``x`` because a
# byte-level decoder cannot restore case the normalizer dropped. Nothing tested
# either direction: whether the default really lower-cases, or whether
# ``lowercase=False`` really preserves case AND survives save/load. The second
# half is the one that could rot silently, since ``__init__`` rebuilds the
# normalizer for a freshly built tokenizer but must NOT overwrite an existing
# one on load.


def test_the_default_lowercase_normalizer_is_lossy_as_documented(sample_text_file):
    """Default lowercase=True folds case; decoding cannot restore it."""
    tokenizer = BPETokenizer.train([sample_text_file], vocab_size=100, min_frequency=1)

    assert tokenizer.decode(tokenizer.encode("THE QUICK")) == "the quick"
    # The documented consequence: round-trip is NOT the identity.
    assert tokenizer.decode(tokenizer.encode("HeLLo")) != "HeLLo"


def test_lowercase_false_preserves_case_through_a_roundtrip(sample_text_file):
    """With lowercase=False the round-trip IS the identity for case."""
    tokenizer = BPETokenizer.train([sample_text_file], vocab_size=100, min_frequency=1, lowercase=False)

    for text in ("Hello World", "HELLO", "HeLLo"):
        assert tokenizer.decode(tokenizer.encode(text)) == text, f"{text!r} did not survive the round-trip"


def test_lowercase_false_survives_save_and_load(sample_text_file, tmp_path):
    """A reloaded tokenizer must keep preserving case.

    ``__init__`` rebuilds the normalizer when it CREATES a tokenizer; on load it
    must leave the deserialized one alone or the setting is silently lost.
    """
    tokenizer = BPETokenizer.train([sample_text_file], vocab_size=100, min_frequency=1, lowercase=False)
    save_path = tmp_path / "cased.json"
    tokenizer.save(str(save_path))

    loaded = BPETokenizer.load(str(save_path))
    assert loaded.decode(loaded.encode("Hello World")) == "Hello World"
    assert loaded.encode("Hello World") == tokenizer.encode("Hello World")


def test_byte_level_bpe_round_trips_non_ascii_text(sample_text_file):
    """Byte-level pre-tokenization must survive characters outside ASCII.

    The BPE vocabulary is built from the byte alphabet, so no input should have
    to fall back to UNK -- a regression here would silently corrupt non-English
    text rather than raise.
    """
    tokenizer = BPETokenizer.train([sample_text_file], vocab_size=100, min_frequency=1, lowercase=False)

    for text in ("émoji 🎉 test", "naïve café", "日本語", "tab\there"):
        assert tokenizer.decode(tokenizer.encode(text)) == text, f"{text!r} did not survive the round-trip"


def test_the_vocabulary_api_is_self_consistent(sample_text_file):
    """``vocab_size``, ``get_vocab()`` and the token ids must agree.

    The real vocabulary can be SMALLER than the requested ``vocab_size`` when
    the corpus cannot fill it, so this pins the reported value to the
    tokenizer's actual one rather than to the training argument.
    """
    tokenizer = BPETokenizer.train([sample_text_file], vocab_size=500, min_frequency=1)

    vocabulary = tokenizer.get_vocab()
    assert tokenizer.vocab_size == tokenizer.tokenizer.get_vocab_size()
    assert len(vocabulary) == tokenizer.vocab_size
    # Ids are dense and start at 0.
    assert sorted(vocabulary.values()) == list(range(tokenizer.vocab_size))
