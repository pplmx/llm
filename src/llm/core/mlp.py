import torch
from torch import nn

from llm.utils.common import get_activation_layer


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) 实现，通常用于Transformer架构的FFN部分

    Args:
        hidden_size: 输入和输出的隐藏维度大小
        intermediate_size: 中间层的维度大小，默认为4倍hidden_size
        activation: 激活函数，可以是"gelu"、"relu"或任何nn.Module激活层
        dropout_p: Dropout概率，默认为0.1
        layer_norm_eps: 层归一化的epsilon值，默认为1e-5
        bias: 是否在线性层中使用偏置，默认为True
        use_layer_norm: 是否使用层归一化，默认为True
        device: 模型所在设备
        dtype: 模型参数的数据类型
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int | None = None,
        activation: str | nn.Module = "gelu",
        dropout_p: float = 0.1,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        use_layer_norm: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else 4 * hidden_size
        self.use_layer_norm = use_layer_norm

        # 第一个线性层: hidden_size -> intermediate_size
        self.fc1 = nn.Linear(hidden_size, self.intermediate_size, bias=bias, device=device, dtype=dtype)

        # 激活函数
        if isinstance(activation, str):
            self.activation = get_activation_layer(activation)()
        else:
            self.activation = activation

        # Dropout层
        self.dropout = nn.Dropout(dropout_p)

        # 第二个线性层: intermediate_size -> hidden_size
        self.fc2 = nn.Linear(self.intermediate_size, hidden_size, bias=bias, device=device, dtype=dtype)

        # 层归一化
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps, device=device, dtype=dtype)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化MLP的权重"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状为 [batch_size, seq_len, hidden_size] 或 [batch_size, hidden_size]

        Returns:
            相同形状的输出张量
        """
        # 保存原始形状
        original_shape = x.shape

        # 如果输入是3D张量，转为2D进行处理
        if len(original_shape) > 2:
            x = x.reshape(-1, original_shape[-1])

        # MLP前向传播
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)

        # 残差连接和层归一化 (如果启用)
        if self.use_layer_norm:
            x = self.layer_norm(x + residual)
        else:
            x = x + residual

        # 恢复原始形状
        if len(original_shape) > 2:
            x = x.reshape(*original_shape)

        return x
