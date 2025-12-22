"""
注意力机制教学版本 - 从基础到高级的实现(完整文档版)
=================================================

本模块系统性地实现了注意力机制的核心组件, 按照由浅入深的教学顺序组织:

1. 基础注意力计算 (attention_score) - 理解核心计算
2. 缩放点积注意力 (scaled_dot_product_attention) - 完整实现
3. 单头自注意力 (SingleHeadAttention) - 模块化封装
4. 多头注意力 (MultiHeadAttention) - 工业级实现

教学特色:
- 每个函数/类都有详细的理论说明和参数文档
- 包含完整的数学公式表示
- 添加了实现注意事项和常见问题
- 丰富的示例说明和类比解释
- 类型提示完整, 便于IDE支持

参考论文:
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Transformers in Vision" (Khan et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------- 第1部分: 基础函数 ----------------------------


def attention_score(query: Tensor, key: Tensor) -> Tensor:
    """
    计算基础注意力分数: Q和K的点积相似度

    数学表示:
        Score(Q, K) = Q · K^T

    参数说明:
        query: 查询张量, 形状为 (..., seq_len_q, dim)
               - seq_len_q: 查询序列长度
               - dim: 每个位置的嵌入维度
        key: 键张量, 形状为 (..., seq_len_k, dim)
             - seq_len_k: 键序列长度(通常与seq_len_q相同)

    返回:
        注意力分数张量, 形状为 (..., seq_len_q, seq_len_k)

    示例:
        >>> q = torch.randn(2, 5, 64)  # 批量2, 查询长度5, 维度64
        >>> k = torch.randn(2, 5, 64)  # 键长度5
        >>> scores = attention_score(q, k)  # 输出形状 (2, 5, 5)

    教学说明:
        1. 这是注意力机制最基础的计算形式
        2. 每个查询位置与所有键位置计算相似度
        3. 结果表示查询和键的匹配程度
    """
    return torch.matmul(query, key.transpose(-2, -1))


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> Tensor:
    """
    完整的缩放点积注意力实现

    数学表示:
        Attention(Q, K, V) = softmax(QK^T/√d_k)V

    参数详解:
        query: 查询张量 (B, N, Sq, D)
               - B: 批次大小
               - N: 注意力头数
               - Sq: 查询序列长度
               - D: 每个头的维度
        key: 键张量 (B, N, Sk, D)
        value: 值张量 (B, N, Sv, D)
        attn_mask: 可选掩码, 类型可以是:
                  - 布尔型: True表示屏蔽
                  - 数值型: 加到注意力分数上
        dropout_p: dropout概率
        is_causal: 是否使用因果掩码(自回归模型使用)
        scale: 自定义缩放因子, 默认使用1/√D

    返回:
        注意力输出张量 (B, N, Sq, D)

    实现步骤:
        1. 计算QK^T并缩放
        2. 应用掩码(因果或自定义)
        3. softmax归一化
        4. dropout正则化
        5. 加权求和

    注意事项:
        - 训练和推理时dropout行为不同
        - 因果掩码会覆盖attn_mask
        - 确保value的序列长度与key匹配

    示例:
        >>> q = torch.randn(2, 8, 10, 64)  # 批次2, 8头, 查询长度10
        >>> k = v = torch.randn(2, 8, 10, 64)
        >>> output = scaled_dot_product_attention(q, k, v, is_causal=True)
    """
    # 1. 计算缩放因子
    head_dim = query.size(-1)
    scale_factor = scale if scale is not None else (head_dim**-0.5)

    # 2. 计算并缩放注意力分数
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor

    # 3. 掩码处理
    if is_causal:
        # 生成上三角因果掩码
        seq_len_q, seq_len_k = attn_scores.size(-2), attn_scores.size(-1)
        causal_mask = torch.triu(torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=query.device), diagonal=1)
        attn_scores = attn_scores.masked_fill(causal_mask, -torch.inf)
    elif attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_scores = attn_scores.masked_fill(attn_mask, -torch.inf)
        else:
            attn_scores = attn_scores + attn_mask

    # 4. softmax归一化
    attn_weights = F.softmax(attn_scores, dim=-1)

    # 5. dropout
    if dropout_p > 0.0 and torch.is_grad_enabled():  # 只在训练时dropout
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # 6. 加权求和
    return torch.matmul(attn_weights, value)


# ---------------------------- 第2部分: 单头注意力 ----------------------------


class SingleHeadAttention(nn.Module):
    """
    单头自注意力模块(合并投影优化版)

    架构说明:
        ┌─────────┐    ┌─────────┐
        │ Input   │ →  │ QKV投影 │ → [Q, K, V]
        └─────────┘    └─────────┘
               ↓
        ┌───────────────────────┐
        │ ScaledDotProductAttn  │
        └───────────────────────┘
               ↓
        ┌─────────┐    ┌─────────┐
        │ Output  │ ←  │ 输出投影 │
        └─────────┘    └─────────┘

    初始化参数:
        hidden_size: 输入/输出的隐藏层维度
        dropout_p: 注意力dropout概率
        is_causal: 是否使用因果注意力
        bias: 是否在线性层使用偏置

    实现细节:
        - 使用合并的QKV投影提高效率
        - 支持因果和非因果模式
        - Xavier初始化

    典型应用场景:
        - 小型语言模型
        - 教学演示场景
        - 需要轻量级注意力的任务
    """

    def __init__(self, hidden_size: int, dropout_p: float = 0.1, is_causal: bool = False, bias: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.is_causal = is_causal

        # 合并投影层
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        self._init_weights()

    def _init_weights(self):
        """参数初始化策略"""
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        if self.qkv_proj.bias is not None:
            nn.init.zeros_(self.qkv_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """
        前向传播流程

        输入:
            x: 输入张量 (B, S, H)
            attn_mask: 可选掩码 (B, S, S)或(B, 1, S, S)

        返回:
            输出张量 (B, S, H)

        计算步骤:
            1. QKV投影和分割
            2. 添加头维度
            3. 计算注意力
            4. 输出投影
        """
        B, S, _ = x.shape
        qkv = self.qkv_proj(x)  # (B, S, 3*H)
        q, k, v = qkv.chunk(3, dim=-1)  # 每个(B, S, H)

        # 添加头维度
        q = q.unsqueeze(1)  # (B, 1, S, H)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

        # 调整掩码形状
        if attn_mask is not None and attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)  # (B, 1, S, S)

        # 计算注意力
        attn_output = scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p, is_causal=self.is_causal
        ).squeeze(1)  # (B, S, H)

        # 输出投影
        return self.out_proj(attn_output)


# ---------------------------- 第3部分: 多头注意力 ----------------------------


class MultiHeadAttention(nn.Module):
    """
    多头注意力模块(工业级实现)

    关键特性:
        - 支持多头并行计算
        - 高效的合并投影
        - 完整的掩码支持
        - 优化的内存布局

    数学表示:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
        where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

    参数说明:
        hidden_size: 必须能被num_heads整除
        num_heads: 建议为2的幂次(优化考虑)
        dropout_p: 注意力dropout概率
        is_causal: 是否自回归
        bias: 是否使用投影偏置

    性能考虑:
        - 合并QKV投影减少内存访问
        - 转置操作优化内存连续性
        - 自动广播掩码
    """

    def __init__(
        self, hidden_size: int, num_heads: int, dropout_p: float = 0.1, is_causal: bool = False, bias: bool = True
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) 必须能被 num_heads ({num_heads}) 整除")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout_p = dropout_p
        self.is_causal = is_causal

        # 投影层
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        self._init_weights()

    def _init_weights(self):
        """参数初始化"""
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        if self.qkv_proj.bias is not None:
            nn.init.zeros_(self.qkv_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """
        前向传播

        输入:
            x: (B, S, H)
            attn_mask: (B, S, S) 或 (B, 1, S, S)

        返回:
            (B, S, H)

        计算图:
            x → qkv_proj → split → reshape → attention → reshape → out_proj
        """
        B, S, _ = x.shape
        qkv = self.qkv_proj(x)  # (B, S, 3*H)
        q, k, v = qkv.chunk(3, dim=-1)  # 每个 (B, S, H)

        # 重塑为多头形式
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, N, S, D)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # 调整掩码
        if attn_mask is not None and attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)  # (B, 1, S, S)

        # 计算注意力
        attn_output = scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p, is_causal=self.is_causal
        )  # (B, N, S, D)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, S, N, D)
        attn_output = attn_output.view(B, S, self.hidden_size)  # (B, S, H)

        # 输出投影
        return self.out_proj(attn_output)
