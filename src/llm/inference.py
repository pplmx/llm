import torch

from llm.models.decoder import DecoderModel
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer


@torch.no_grad()
def generate(
    model: DecoderModel,
    tokenizer: SimpleCharacterTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> str:
    """
    接收 prompt, 并使用训练好的模型生成文本.

    参数:
        model: 训练好的解码器模型.
        tokenizer: 用于编码和解码文本的分词器.
        prompt: 输入的文本提示.
        max_new_tokens: 要生成的最大 token 数量.
        temperature: 控制生成文本的随机性. 值越小, 生成的文本越确定. 为 0 时使用 Greedy Search.
        top_k: Top-k 采样. 如果不为 None, 则从概率最高的 k 个 token 中采样.
    """
    model.eval()
    device = next(model.parameters()).device

    # 1. 将 prompt 转换为 token ids
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    # 获取模型限制
    max_seq_len = getattr(model, "max_seq_len", 512)

    # 用于缓存及生成的 token
    generated_tokens = input_ids.copy()
    past_key_values = None

    # 第一步: 预热 (Prefill)
    # 如果 prompt 超过 max_seq_len, 截断它
    if input_tensor.size(1) > max_seq_len:
        input_tensor = input_tensor[:, -max_seq_len:]

    # 获取第一个输出并初始化 KV Cache
    logits, past_key_values = model(input_tensor, use_cache=True)
    next_token_logits = logits[0, -1, :]

    # 自回归生成循环
    for _ in range(max_new_tokens):
        # 采样策略
        if temperature == 0:
            # Greedy Search
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            # Sampling
            next_token_logits = next_token_logits / temperature
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[..., -1, None]] = -float("Inf")

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        # 添加到生成序列
        token_id = int(next_token.item())
        generated_tokens.append(token_id)

        # 下一步输入仅为新生成的 token (Incremental decoding)
        next_input = next_token.unsqueeze(0)
        logits, past_key_values = model(next_input, past_key_values=past_key_values, use_cache=True)
        next_token_logits = logits[0, -1, :]

    # 将生成的 token ids 解码回文本
    return tokenizer.decode(generated_tokens)


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
