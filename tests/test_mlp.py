import torch

from llm.core.mlp import MLP


def test_mlp():
    """测试MLP模块的功能"""
    torch.manual_seed(42)

    # 配置参数
    seq_len = 512
    batch_size = 2
    hidden_size = 128
    ffn_hidden_size = 4 * hidden_size
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建输入张量
    input_tensor = torch.rand((batch_size, seq_len, hidden_size), dtype=dtype, device=device)

    # 初始化MLP模型
    mlp = MLP(
        hidden_size=hidden_size,
        intermediate_size=ffn_hidden_size,
        activation="gelu",
        dropout_p=0.1,
        use_layer_norm=True,
        device=device,
        dtype=dtype,
    )

    # 前向传播
    with torch.no_grad():
        output = mlp(input_tensor)

    # 验证输出形状
    assert output.shape == input_tensor.shape, f"输出形状 {output.shape} 与输入形状 {input_tensor.shape} 不匹配"

    # 返回结果以供进一步检查
    return {
        "input": input_tensor.detach(),
        "output": output.detach(),
    }
