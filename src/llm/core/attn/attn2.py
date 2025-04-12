import math

import torch
from torch import nn


class AttentionBlock(nn.Module):
    """
    基础注意力块，使用 nn.Sequential 实现内部结构

    Args:
        hidden_size: 隐藏层维度大小
        dropout_p: Dropout 概率, 默认为 0.1
        bias: 是否在线性层中使用偏置, 默认为 True
        device: 模型所在设备
        dtype: 模型参数的数据类型
    """

    def __init__(
        self,
        hidden_size: int,
        dropout_p: float = 0.1,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.scaling = math.sqrt(hidden_size)

        # 使用 nn.Sequential 组织 QKV 投影
        self.qkv_proj = nn.Sequential(
            nn.Linear(hidden_size, 3 * hidden_size, bias=bias, device=device, dtype=dtype),
        )

        # 输出投影
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype), nn.Dropout(dropout_p)
        )

        self._init_weights()

    def _init_weights(self):
        """初始化注意力层的权重"""
        # 获取 QKV 投影中的线性层
        qkv_linear = self.qkv_proj[0]
        nn.init.xavier_uniform_(qkv_linear.weight)
        if qkv_linear.bias is not None:
            nn.init.zeros_(qkv_linear.bias)

        # 获取输出投影中的线性层
        out_linear = self.out_proj[0]
        nn.init.xavier_uniform_(out_linear.weight)
        if out_linear.bias is not None:
            nn.init.zeros_(out_linear.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        注意力机制的前向传播

        Args:
            hidden_states: 输入张量, 形状为 [batch_size, seq_len, hidden_size]
            attention_mask: 可选的掩码张量, 形状为 [batch_size, 1, 1, seq_len] 或
                           [batch_size, 1, seq_len, seq_len]

        Returns:
            形状为 [batch_size, seq_len, hidden_size] 的输出张量
        """
        batch_size, seq_len, _ = hidden_states.size()

        # 投影获取 Q, K, V
        qkv = self.qkv_proj(hidden_states)  # [batch_size, seq_len, 3 * hidden_size]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.hidden_size)
        qkv = qkv.permute(2, 0, 1, 3)  # [3, batch_size, seq_len, hidden_size]
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分离 Q, K, V

        # 计算缩放点积注意力
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / self.scaling

        # 应用注意力掩码（如果提供）
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # 归一化注意力分数为概率
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

        # 计算加权和
        context_layer = torch.matmul(attention_probs, v)

        # 通过输出投影
        output = self.out_proj(context_layer)

        return output


class MultiHeadAttentionBlock(nn.Module):
    """
    使用 nn.Sequential 组织的多头注意力块

    Args:
        hidden_size: 隐藏层维度大小
        num_attention_heads: 注意力头的数量, 默认为 8
        dropout_p: Dropout 概率, 默认为 0.1
        bias: 是否在线性层中使用偏置, 默认为 True
        layer_norm_eps: Layer Normalization 的 epsilon 值, 默认为 1e-5
        use_layer_norm: 是否使用 Layer Normalization, 默认为 True
        device: 模型所在设备
        dtype: 模型参数的数据类型
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 8,
        dropout_p: float = 0.1,
        bias: bool = True,
        layer_norm_eps: float = 1e-5,
        use_layer_norm: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        assert hidden_size % num_attention_heads == 0, "隐藏层大小必须能被注意力头数量整除"

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_size = hidden_size // num_attention_heads
        self.scaling = math.sqrt(self.head_size)
        self.use_layer_norm = use_layer_norm

        # 使用 Sequential 组织层归一化和投影层
        layers = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_size, eps=layer_norm_eps, device=device, dtype=dtype))

        # 组织 QKV 投影
        self.qkv_proj = nn.Sequential(
            *layers,
            nn.Linear(hidden_size, 3 * hidden_size, bias=bias, device=device, dtype=dtype),
        )

        # 输出投影
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype), nn.Dropout(dropout_p)
        )

        self._init_weights()

    def _init_weights(self):
        """初始化多头注意力层的权重"""
        # 获取 QKV 投影中的线性层（根据是否有 LayerNorm，可能在不同位置）
        qkv_linear = self.qkv_proj[-1]
        nn.init.xavier_uniform_(qkv_linear.weight)
        if qkv_linear.bias is not None:
            nn.init.zeros_(qkv_linear.bias)

        # 获取输出投影中的线性层
        out_linear = self.out_proj[0]
        nn.init.xavier_uniform_(out_linear.weight)
        if out_linear.bias is not None:
            nn.init.zeros_(out_linear.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        多头注意力机制的前向传播

        Args:
            hidden_states: 输入张量, 形状为 [batch_size, seq_len, hidden_size]
            attention_mask: 可选的掩码张量, 形状为 [batch_size, 1, 1, seq_len] 或
                           [batch_size, 1, seq_len, seq_len]

        Returns:
            形状为 [batch_size, seq_len, hidden_size] 的输出张量
        """
        # 保存残差连接
        residual = hidden_states

        batch_size, seq_len, _ = hidden_states.size()

        # 投影获取 Q, K, V
        qkv = self.qkv_proj(hidden_states)  # [batch_size, seq_len, 3 * hidden_size]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_attention_heads, self.head_size)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_size]
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分离 Q, K, V

        # 计算缩放点积注意力
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / self.scaling

        # 应用注意力掩码（如果提供）
        if attention_mask is not None:
            # 扩展 mask 以适配多头
            attention_scores = attention_scores + attention_mask

        # 归一化注意力分数为概率
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

        # 计算加权和
        context_layer = torch.matmul(attention_probs, v)  # [batch_size, num_heads, seq_len, head_size]

        # 重塑回原始维度
        context_layer = context_layer.transpose(1, 2).contiguous()  # [batch_size, seq_len, num_heads, head_size]
        context_layer = context_layer.reshape(batch_size, seq_len, self.hidden_size)

        # 通过输出投影
        output = self.out_proj(context_layer)

        # 添加残差连接
        output = residual + output

        return output


class TransformerEncoderLayer(nn.Module):
    """
    使用 nn.Sequential 组织的 Transformer 编码层

    包含一个多头注意力块和一个前馈网络块

    Args:
        hidden_size: 隐藏层维度大小
        num_attention_heads: 注意力头的数量, 默认为 8
        intermediate_size: 前馈网络中间层的大小, 默认为 None (将设置为 4 * hidden_size)
        activation: 激活函数, 可以是 "gelu", "relu" 或任何 nn.Module 激活层, 默认为 "gelu"
        dropout_p: Dropout 概率, 默认为 0.1
        layer_norm_eps: Layer Normalization 的 epsilon 值, 默认为 1e-5
        bias: 是否在线性层中使用偏置, 默认为 True
        device: 模型所在设备
        dtype: 模型参数的数据类型
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 8,
        intermediate_size: int | None = None,
        activation: str | nn.Module = "gelu",
        dropout_p: float = 0.1,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # 设置默认的中间层大小
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size

        # 多头注意力块
        self.attention = MultiHeadAttentionBlock(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            dropout_p=dropout_p,
            bias=bias,
            layer_norm_eps=layer_norm_eps,
            use_layer_norm=True,  # Pre-LN 架构
            device=device,
            dtype=dtype,
        )

        # 获取激活函数
        if isinstance(activation, str):
            if activation == "gelu":
                act_fn = nn.GELU()
            elif activation == "relu":
                act_fn = nn.ReLU()
            else:
                raise ValueError(f"不支持的激活函数: {activation}")
        else:
            act_fn = activation

        # 前馈网络块
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_size, eps=layer_norm_eps, device=device, dtype=dtype),
            nn.Linear(hidden_size, intermediate_size, bias=bias, device=device, dtype=dtype),
            act_fn,
            nn.Dropout(dropout_p),
            nn.Linear(intermediate_size, hidden_size, bias=bias, device=device, dtype=dtype),
            nn.Dropout(dropout_p),
        )

        self._init_weights()

    def _init_weights(self):
        """初始化前馈网络的权重"""
        # FFN中的线性层初始化
        ffn_fc1 = self.ffn[1]  # 第一个线性层
        ffn_fc2 = self.ffn[4]  # 第二个线性层

        nn.init.xavier_uniform_(ffn_fc1.weight)
        nn.init.xavier_uniform_(ffn_fc2.weight)

        if ffn_fc1.bias is not None:
            nn.init.zeros_(ffn_fc1.bias)
        if ffn_fc2.bias is not None:
            nn.init.zeros_(ffn_fc2.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Transformer 编码层的前向传播

        Args:
            hidden_states: 输入张量, 形状为 [batch_size, seq_len, hidden_size]
            attention_mask: 可选的注意力掩码张量

        Returns:
            形状为 [batch_size, seq_len, hidden_size] 的输出张量
        """
        # 多头注意力层
        hidden_states = self.attention(hidden_states, attention_mask)

        # 保存残差连接
        residual = hidden_states

        # 前馈网络
        ffn_output = self.ffn(hidden_states)

        # 添加残差连接
        output = residual + ffn_output

        return output


class TransformerEncoder(nn.Module):
    """
    使用 nn.Sequential 组织的完整 Transformer 编码器

    由多个 TransformerEncoderLayer 组成

    Args:
        hidden_size: 隐藏层维度大小
        num_hidden_layers: 编码层的数量, 默认为 6
        num_attention_heads: 注意力头的数量, 默认为 8
        intermediate_size: 前馈网络中间层的大小, 默认为 None (将设置为 4 * hidden_size)
        activation: 激活函数, 可以是 "gelu", "relu" 或任何 nn.Module 激活层, 默认为 "gelu"
        dropout_p: Dropout 概率, 默认为 0.1
        layer_norm_eps: Layer Normalization 的 epsilon 值, 默认为 1e-5
        bias: 是否在线性层中使用偏置, 默认为 True
        device: 模型所在设备
        dtype: 模型参数的数据类型
    """

    def __init__(
        self,
        hidden_size: int,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        intermediate_size: int | None = None,
        activation: str | nn.Module = "gelu",
        dropout_p: float = 0.1,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # 创建多个 Transformer 编码层
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    intermediate_size=intermediate_size,
                    activation=activation,
                    dropout_p=dropout_p,
                    layer_norm_eps=layer_norm_eps,
                    bias=bias,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_hidden_layers)
            ]
        )

        # 最终的层归一化
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps, device=device, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Transformer 编码器的前向传播

        Args:
            hidden_states: 输入张量, 形状为 [batch_size, seq_len, hidden_size]
            attention_mask: 可选的注意力掩码张量

        Returns:
            形状为 [batch_size, seq_len, hidden_size] 的输出张量
        """
        # 依次通过每个编码层
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # 最终的层归一化
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


# 如何使用这些模块的示例
def example_usage():
    # 创建一个简单的 Transformer 编码器
    encoder = TransformerEncoder(
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        dropout_p=0.1,
    )

    # 创建示例输入
    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(batch_size, seq_len, 512)

    # 创建注意力掩码（可选）
    # 1表示要关注的位置，0表示要屏蔽的位置
    attention_mask = torch.ones(batch_size, 1, 1, seq_len)

    # 前向传播
    output = encoder(hidden_states, attention_mask)

    return output
