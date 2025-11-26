import torch

from llm.models.decoder import DecoderModel
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer


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
        temperature: 控制生成文本的随机性. 值越小, 生成的文本越确定.
        top_k: Top-k 采样. 如果不为 None, 则从概率最高的 k 个 token 中采样.
    """
    model.eval()

    # 1. 将 prompt 转换为 token ids
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # 添加 batch 维度

    # 获取设备信息
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # 用于存储生成的 token
    generated_tokens = input_ids.copy()

    with torch.no_grad():
        # 自回归生成循环
        for _ in range(max_new_tokens):
            # 截取模型上下文长度内的 tokens (默认使用512作为最大序列长度)
            seq_length = input_tensor.size(1)
            max_seq_len = 512  # 默认值，与模型中一致
            start_pos = max(0, seq_length - max_seq_len)
            context_tensor = input_tensor[:, start_pos:]

            # 获取模型输出 logits
            logits = model(context_tensor)

            # 只关注最后一个时间步的 logits
            next_token_logits = logits[0, -1, :] / temperature

            # 应用 top-k 过滤
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("inf"))

            # 计算概率分布
            probs = torch.softmax(next_token_logits, dim=-1)

            # 采样下一个 token
            next_token = torch.multinomial(probs, num_samples=1)

            # 添加到生成序列
            generated_tokens.append(int(next_token.item()))

            # 更新 input_tensor 用于下一次迭代
            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)

    # 将生成的 token ids 解码回文本
    generated_text = tokenizer.decode(generated_tokens)

    return generated_text


if __name__ == "__main__":
    # 这是一个如何使用 generate 函数的示例

    # 1. 准备一个简单的语料库来初始化分词器
    corpus = ["hello world!", "this is a test.", "你好 世界！"]
    tokenizer = SimpleCharacterTokenizer(corpus)
    print(f"分词器词汇表大小: {tokenizer.vocab_size}")

    # 2. 初始化模型 (使用与分词器匹配的 vocab_size)
    # 注意: 这里的超参数是示例, 实际应与您训练时使用的参数匹配
    model = DecoderModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
    )
    print("模型已初始化 (未加载权重).")

    # 3. (可选) 加载训练好的权重
    # try:
    #     checkpoint = torch.load("path/to/your/checkpoint.pth")
    #     model.load_state_dict(checkpoint)
    #     print("模型权重已加载.")
    # except FileNotFoundError:
    #     print("警告: 未找到模型权重文件, 将使用随机初始化的模型.")

    # 4. 运行生成
    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="hello",
        max_new_tokens=10,
    )
    print(f"\n生成结果: {generated_text}")
