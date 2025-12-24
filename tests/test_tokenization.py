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

    assert tokenizer.vocab_size > 0
    assert tokenizer.pad_token_id is not None

    # Encode / Decode
    text = "hello world"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text

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
