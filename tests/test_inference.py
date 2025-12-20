import pytest

from llm.inference import generate
from llm.models.decoder import DecoderModel
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer


@pytest.fixture
def model_and_tokenizer():
    corpus = ["hello world!", "this is a test.", "你好 世界！"]
    tokenizer = SimpleCharacterTokenizer(corpus)
    model = DecoderModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_layers=1,
        num_heads=2,
        max_seq_len=64,
    )
    return model, tokenizer


def test_generate_greedy(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompt = "hello"
    max_new_tokens = 5

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0,
    )

    assert isinstance(output, str)
    assert len(output) > len(prompt)
    # 验证长度: SimpleCharacterTokenizer 每个字符是一个 token
    # 生成的文本应包含原始 prompt 和生成的 tokens
    # 注意: SimpleCharacterTokenizer 解码时可能包含 prompt 以外的字符
    encoded_prompt = tokenizer.encode(prompt)
    encoded_output = tokenizer.encode(output)
    assert len(encoded_output) == len(encoded_prompt) + max_new_tokens


def test_generate_sampling(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompt = "test"
    max_new_tokens = 3

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=2,
    )

    assert isinstance(output, str)
    encoded_prompt = tokenizer.encode(prompt)
    encoded_output = tokenizer.encode(output)
    assert len(encoded_output) == len(encoded_prompt) + max_new_tokens


def test_generate_max_seq_len_truncation(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    # 创建一个超过 max_seq_len 的 prompt
    prompt = "a" * 100
    max_new_tokens = 2

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0,
    )

    # 验证是否正常运行且返回合理的长度
    assert isinstance(output, str)
    # 注意: generate 函数目前返回的是 generated_tokens 解码后的结果
    # 其中的 generated_tokens 初始化为 input_ids.copy()
    # 即使 input_tensor 被截断，generated_tokens 仍然保留了完整输入
    encoded_output = tokenizer.encode(output)
    assert len(encoded_output) == 100 + max_new_tokens
