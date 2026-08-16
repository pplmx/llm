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
    """Regression (RIL ISS-155): ``pad_token_id`` returned
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
