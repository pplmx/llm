from collections.abc import Generator

import torch

from llm.models.decoder import DecoderModel
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer


@torch.no_grad()
def stream_generate(
    model: DecoderModel,
    tokenizer: SimpleCharacterTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
) -> Generator[str]:
    """
    逐步生成文本的生成器函数.

    yields:
        str: 新生成的文本片段 (通常是一个 token 解码后的字符).
    """
    model.eval()
    device = next(model.parameters()).device

    # 1. 将 prompt 转换为 token ids
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    # 获取模型限制
    max_seq_len = getattr(model, "max_seq_len", 512)

    past_key_values = None

    # 第一步: 预热 (Prefill)
    # 如果 prompt + max_new_tokens 超过 max_seq_len, 截断它以留出空间
    if input_tensor.size(1) + max_new_tokens > max_seq_len:
        input_tensor = input_tensor[:, -(max_seq_len - max_new_tokens) :]

    # 获取第一个输出并初始化 KV Cache
    logits, past_key_values = model(input_tensor, use_cache=True)
    next_token_logits = logits[0, -1, :]

    # 避免生成 PAD token
    if hasattr(tokenizer, "pad_token_id"):
        next_token_logits[tokenizer.pad_token_id] = -float("inf")

    # 跟踪生成历史用于重复惩罚
    # 初始化为 input_ids (prompt)
    generated_ids = input_ids.copy()

    # 自回归生成循环
    for _ in range(max_new_tokens):
        # 1. 重复惩罚 (Repetition Penalty)
        # 论文: https://arxiv.org/abs/1909.05858
        if repetition_penalty != 1.0:
            # 创建一个 scatter 掩码
            score = torch.gather(next_token_logits, 0, torch.tensor(generated_ids, device=device))

            # 如果 score < 0, 惩罚是乘以 penalty (加大负值)
            # 如果 score > 0, 惩罚是除以 penalty (减小正值)
            score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)

            next_token_logits.scatter_(0, torch.tensor(generated_ids, device=device), score)

        # 采样策略
        if temperature == 0:
            # Greedy Search
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            # Sampling
            next_token_logits = next_token_logits / temperature

            # Top-K
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[..., -1, None]] = -torch.inf

            # Top-P (Nucleus Sampling)
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float("inf")

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        # 获取 token id
        token_id = int(next_token.item())
        generated_ids.append(token_id)

        # 解码当前 token 并 yield
        # 注意: 这里简单地解码单个 token, 对于 BPE 等可能需要处理字节碎片,
        # 但对于 SimpleCharacterTokenizer 是安全的.
        text_chunk = tokenizer.decode([token_id])
        yield text_chunk

        # 下一步输入仅为新生成的 token (Incremental decoding)
        next_input = next_token.unsqueeze(0)
        logits, past_key_values = model(next_input, past_key_values=past_key_values, use_cache=True)
        next_token_logits = logits[0, -1, :]

        # 避免生成 PAD token
        if hasattr(tokenizer, "pad_token_id"):
            next_token_logits[tokenizer.pad_token_id] = -float("inf")


def generate(
    model: DecoderModel,
    tokenizer: SimpleCharacterTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
) -> str:
    """
    接收 prompt, 并使用训练好的模型生成文本.
    """
    # 使用 stream_generate 并拼接所有输出
    generator = stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    return prompt + "".join(list(generator))


if __name__ == "__main__":
    # 这是一个如何使用 generate 函数的示例

    # 1. 准备一个简单的语料库来初始化分词器
    corpus = ["hello world!", "this is a test.", "你好 世界!"]
    tokenizer = SimpleCharacterTokenizer(corpus)
    print(f"分词器词汇表大小: {tokenizer.vocab_size}")

    # 2. 初始化模型 (使用与分词器匹配的 vocab_size)
    model = DecoderModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        max_seq_len=128,
    )
    print("模型已初始化 (未加载权重).")

    # 3. 运行生成
    print("\n测试 Greedy Search (temperature=0):")
    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="hello",
        max_new_tokens=10,
        temperature=0,
    )
    print(f"生成结果: {generated_text}")

    print("\n测试 Sampling (temperature=0.8, top_k=5):")
    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="hello",
        max_new_tokens=10,
        temperature=0.8,
        top_k=5,
    )
    print(f"生成结果: {generated_text}")
