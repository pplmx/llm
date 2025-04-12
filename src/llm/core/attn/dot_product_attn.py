import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def scaled_dot_product_attn(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale_factor: float = 1.0,
) -> Tensor:
    """
    计算缩放点积注意力。

    参数:
      query: Tensor，形状 (batch_size, num_heads, seq_len_q, head_dim)
      key:   Tensor，形状 (batch_size, num_heads, seq_len_k, head_dim)
      value: Tensor，形状 (batch_size, num_heads, seq_len_k, head_dim)
      attn_mask: 可选的注意力 mask，广播到 (batch_size, num_heads, seq_len_q, seq_len_k)，mask 为 True 的位置被置为 -inf
      dropout_p: float，注意力权重 dropout 概率
      is_causal: bool，是否应用因果 mask（防止信息泄露）
      scale_factor: float，缩放因子，通常为 sqrt(head_dim)

    返回:
      Tensor，形状 (batch_size, num_heads, seq_len_q, head_dim)
    """
    # 计算点积注意力分数
    scores = torch.matmul(query, key.transpose(-2, -1)) / scale_factor

    # 如果提供了 attn_mask，则将对应位置置为 -inf
    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask, float("-inf"))

    # 应用因果 mask，保证后续时刻不能影响前面的信息
    if is_causal:
        seq_len_q, seq_len_k = scores.size(-2), scores.size(-1)
        causal_mask = torch.tril(torch.ones((seq_len_q, seq_len_k), device=scores.device)).bool()
        scores = scores.masked_fill(~causal_mask, float("-inf"))

    # 计算 softmax 得到归一化的注意力权重，并应用 dropout（如果需要）
    attn_weights = F.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

    # 计算注意力输出
    output = torch.matmul(attn_weights, value)
    return output


class SelfAttention(nn.Module):
    """
    单头自注意力模块
    """

    def __init__(self, hidden_size: int, dropout: float = 0.0, is_causal: bool = False):
        """
        参数:
           hidden_size: 输入的隐藏状态维度，同时也是 Q、K、V 的维度
           dropout: dropout 概率
           is_causal: 是否应用因果 mask
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.is_causal = is_causal

        # 线性映射：生成 Q、K、V
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        # 输出投影
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """
        参数:
          hidden_states: Tensor，形状 (batch_size, seq_len, hidden_size)
          attn_mask: 可选注意力 mask，形状 (batch_size, seq_len, seq_len) 或可广播的形状
        返回:
          Tensor，形状 (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 得到 query, key, value（单头实现）
        q = self.q_proj(hidden_states)  # (batch_size, seq_len, hidden_size)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # 增加 head 维度，视作单头 (batch_size, 1, seq_len, hidden_size)
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

        scaling = math.sqrt(self.hidden_size)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)

        attn_output = scaled_dot_product_attn(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
            is_causal=self.is_causal,
            scale_factor=scaling,
        )  # (batch_size, 1, seq_len, hidden_size)

        # 去除 head 维度并进行输出投影
        attn_output = attn_output.squeeze(1)  # (batch_size, seq_len, hidden_size)
        out = self.out_proj(attn_output)
        return out


class MultiHeadAttention(nn.Module):
    """
    多头注意力模块
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0, is_causal: bool = False):
        """
        参数:
           hidden_size: 输入的隐藏状态维度
           num_heads: 注意力头的数量，要求 hidden_size 能被 num_heads 整除
           dropout: dropout 概率
           is_causal: 是否应用因果 mask（如在语言模型中）
        """
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size 必须能被 num_heads 整除")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.is_causal = is_causal
        self.head_dim = hidden_size // num_heads

        # 生成 q, k, v 的线性映射
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        # 输出投影
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """
        参数:
          hidden_states: Tensor，形状 (batch_size, seq_len, hidden_size)
          attn_mask: 可选注意力 mask，形状 (batch_size, seq_len, seq_len) 或可广播的形状
        返回:
          Tensor，形状 (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 生成 query, key, value
        q = self.q_proj(hidden_states)  # (batch_size, seq_len, hidden_size)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # 将 hidden_size 分解为多个头：形状变为 (batch_size, seq_len, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 转置为 (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scaling = math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)

        attn_output = scaled_dot_product_attn(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
            is_causal=self.is_causal,
            scale_factor=scaling,
        )  # (batch_size, num_heads, seq_len, head_dim)

        # 拼接多头输出 (batch_size, seq_len, num_heads * head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # 输出投影
        out = self.out_proj(attn_output)
        return out


# -------------------------
# 示例使用
if __name__ == "__main__":
    # 假设 batch_size = 2, 序列长度 = 5, 隐藏状态维度 = 16, heads = 4
    batch_size, seq_len, hidden_size, num_heads = 2, 5, 16, 4
    dummy_hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # 单头自注意力模块
    self_attn = SelfAttention(hidden_size, dropout=0.1, is_causal=False)
    out_self_attn = self_attn(dummy_hidden_states)
    print("Self-Attention 输出形状:", out_self_attn.shape)  # (batch_size, seq_len, hidden_size)

    # 多头注意力模块
    multi_head_attn = MultiHeadAttention(hidden_size, num_heads=num_heads, dropout=0.1, is_causal=False)
    out_multi_attn = multi_head_attn(dummy_hidden_states)
    print("Multi-Head Attention 输出形状:", out_multi_attn.shape)  # (batch_size, seq_len, hidden_size)
