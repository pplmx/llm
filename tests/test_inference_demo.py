"""
Inference Demo Tests

Tests text generation with greedy search and sampling.
"""

from llm.inference import generate
from llm.models.decoder import DecoderModel
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer


def test_inference_greedy_search():
    """Test greedy search generation."""
    corpus = ["hello world!", "this is a test.", "你好 世界!"]
    tokenizer = SimpleCharacterTokenizer(corpus)

    model = DecoderModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        max_seq_len=128,
    )
    model.eval()

    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="hello",
        max_new_tokens=10,
        temperature=0,
    )

    assert isinstance(generated_text, str)
    assert len(generated_text) > len("hello")
    assert generated_text.startswith("hello")


def test_inference_sampling():
    """Test sampling generation with temperature and top_k."""
    corpus = ["hello world!", "this is a test.", "你好 世界!"]
    tokenizer = SimpleCharacterTokenizer(corpus)

    model = DecoderModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        max_seq_len=128,
    )
    model.eval()

    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="hello",
        max_new_tokens=10,
        temperature=0.8,
        top_k=5,
    )

    assert isinstance(generated_text, str)
    assert len(generated_text) > len("hello")
