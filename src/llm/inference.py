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

    # TODO: 实现自回归生成循环
    # TODO: 实现 KV 缓存优化
    # TODO: 实现 top-k / top-p 采样
    # TODO: 将生成的 token ids 解码回文本
    print(f"输入 prompt '{prompt}' 已被编码为: {input_ids}")

    return "文本生成功能待实现..."


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
