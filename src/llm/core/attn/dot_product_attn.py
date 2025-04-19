"""
注意力机制教学版本 - 从基础到高级的实现
====================================

本模块实现了注意力机制的几个关键版本，按照复杂度由浅入深:
1. 基础注意力计算 (attention_score)
2. 缩放点积注意力 (scaled_dot_product_attention)
3. 多头注意力 (MultiHeadAttention)

这种结构设计使得初学者可以循序渐进地理解注意力机制的原理和实现细节。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ------------------------------------ 第1部分: 基础函数 ------------------------------------


def attention_score(query: Tensor, key: Tensor) -> Tensor:
    """
    计算基础注意力分数: Q @ K^T。

    参数:
        query (Tensor): 查询张量, 形状 (..., seq_len_q, dim)。
        key (Tensor): 键张量, 形状 (..., seq_len_k, dim)。

    返回:
        Tensor: 注意力分数, 形状 (..., seq_len_q, seq_len_k)。
    """
    # 矩阵乘法 Q * K^T
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
    计算缩放点积注意力 (Scaled Dot-Product Attention)。

    参数:
        query (Tensor): 查询张量, 形状 (B, N, Sq, D)。 B=批次, N=头数, Sq=查询序列长度, D=头维度。
        key (Tensor): 键张量, 形状 (B, N, Sk, D)。 Sk=键序列长度。
        value (Tensor): 值张量, 形状 (B, N, Sv, D)。 通常 Sk == Sv。
        attn_mask (Optional[Tensor]): 注意力掩码。
            - 若为布尔张量, `True` 表示屏蔽该位置。需能广播至 (B, N, Sq, Sk)。
            - 若为浮点张量, 会加到注意力分数上 (屏蔽位应为 -inf)。
            默认为 None。
        dropout_p (float): 应用于注意力权重的 dropout 概率。默认为 0.0。
        is_causal (bool): 若为 True, 应用因果掩码 (上三角掩码)，阻止关注未来位置。
                          若为 True, 会忽略 attn_mask。默认为 False。
        scale (Optional[float]): 缩放因子。若为 None, 则使用 1 / sqrt(head_dim)。默认为 None。

    返回:
        Tensor: 注意力输出张量, 形状 (B, N, Sq, D)。
    """
    # 1. 获取头维度并计算缩放因子
    head_dim = query.size(-1)
    # 若未提供 scale，则使用 1/sqrt(d_k) 作为默认值
    scale = scale or head_dim**-0.5

    # 2. 计算注意力分数并应用缩放: (Q @ K^T) * scale
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    # 3. 应用掩码 (优先处理因果掩码)
    if is_causal:
        # 创建因果掩码 (上三角矩阵, 对角线为 0)
        seq_len_q, seq_len_k = attn_scores.shape[-2:]
        # True 表示需要屏蔽的位置 (不允许关注未来)
        causal_mask = torch.triu(
            torch.ones((seq_len_q, seq_len_k), device=query.device, dtype=torch.bool),
            diagonal=1,  # diagonal=1 屏蔽对角线及以上，但注意力中通常允许看自己，所以是 > i 的位置
        )
        # 使用 -inf 填充被屏蔽的分数
        attn_scores = attn_scores.masked_fill(causal_mask, -torch.inf)
    elif attn_mask is not None:  # 如果不是因果且提供了 attn_mask
        if attn_mask.dtype == torch.bool:
            attn_scores = attn_scores.masked_fill(attn_mask, -torch.inf)
        else:  # 假设是加性掩码
            attn_scores = attn_scores + attn_mask

    # 4. 计算注意力权重: softmax(scores)
    attn_weights = F.softmax(attn_scores, dim=-1)  # 在 key 维度上进行 softmax

    # 5. 应用 Dropout (仅在 dropout_p > 0 时)
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)  # 注意：训练/评估模式由调用者控制

    # 6. 计算加权值: weights @ V
    output = torch.matmul(attn_weights, value)

    return output


# ------------------------------------ 第2部分: 多头注意力 ------------------------------------


class MultiHeadAttention(nn.Module):
    """
    多头注意力模块 (优化: 使用合并的 QKV 投影)。
    """

    def __init__(self, hidden_size: int, num_heads: int, p: float = 0.0, is_causal: bool = False, bias: bool = True):
        """
        参数:
            hidden_size (int): 输入/输出的隐藏状态维度。
            num_heads (int): 注意力头的数量。
            p (float): 注意力权重的 Dropout 概率。
            is_causal (bool): 是否使用因果掩码。
            bias (bool): 线性层是否使用偏置。
        """
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"隐藏维度 hidden_size ({hidden_size}) 必须能被头数 num_heads ({num_heads}) 整除")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # 每个头的维度
        self.p = p  # 注意力内部 dropout 概率
        self.is_causal = is_causal

        # 合并的 Q, K, V 线性投影层
        self.in_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        # 输出投影层
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        # 输出 dropout 层
        self.dropout = nn.Dropout(p)

        self._init_weights()

    def _init_weights(self):
        """初始化权重 (Xavier uniform) 和偏置 (zeros)"""
        nn.init.xavier_uniform_(self.in_proj.weight)
        if self.in_proj.bias is not None:
            nn.init.zeros_(self.in_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, hidden_states: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """
        前向传播。

        参数:
            hidden_states (Tensor): 输入隐藏状态 [B, S, H]。 B=批次, S=序列长度, H=隐藏维度。
            attn_mask (Optional[Tensor]): 注意力掩码 [B, S, S] 或 [B, 1, Sq, Sk]。

        返回:
            Tensor: 注意力层输出 [B, S, H]。
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 1. 计算 Q, K, V 投影并分割
        qkv = self.in_proj(hidden_states)  # [B, S, 3*H]
        query, key, value = qkv.chunk(3, dim=-1)  # 每个都是 [B, S, H]

        # 2. 重塑 Q, K, V 以适配多头: [B, S, H] -> [B, N, S, D]
        #   - H = N * D (隐藏维度 = 头数 * 头维度)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 调整掩码维度 (如果需要)
        if attn_mask is not None:
            # 确保 mask 形状是 [B, N, Sq, Sk] 或可广播
            if attn_mask.dim() == 3:  # e.g., [B, S, S] -> [B, 1, S, S] (将自动广播到 N 头)
                attn_mask = attn_mask.unsqueeze(1)
            # 如果已经是 [B, 1, S, S] 或 [B, N, S, S] 则无需处理

        # 4. 计算缩放点积注意力
        attn_output = scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=self.p if self.training else 0.0,  # 训练时应用 dropout
            is_causal=self.is_causal,
            scale=None,  # 让函数内部计算 scale = 1/sqrt(head_dim)
        )  # 输出 [B, N, S, D]

        # 5. 重塑输出: [B, N, S, D] -> [B, S, N, D] -> [B, S, H]
        # contiguous() 确保存储连续，以便 view 操作
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        # 6. 应用输出投影和 dropout
        output = self.dropout(self.out_proj(attn_output))  # [B, S, H]

        return output
