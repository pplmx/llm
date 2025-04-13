"""
注意力机制教学版本 - 从基础到高级的实现
====================================

本模块实现了注意力机制的几个关键版本，按照复杂度由浅入深:
1. 基础注意力计算 (attention_score)
2. 缩放点积注意力 (scaled_dot_product_attention)
3. 单头自注意力 (SingleHeadAttention)
4. 多头注意力 (MultiHeadAttention)

这种结构设计使得初学者可以循序渐进地理解注意力机制的原理和实现细节。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ------------------------------------ 第1部分: 基础函数 ------------------------------------


def attention_score(query: Tensor, key: Tensor) -> Tensor:
    """
    计算最基本的注意力分数 - 点积

    参数:
        query: 形状为 (..., seq_len_q, dim)
        key: 形状为 (..., seq_len_k, dim)

    返回:
        注意力分数，形状为 (..., seq_len_q, seq_len_k)
    """
    # 计算点积，返回的形状是 (..., seq_len_q, seq_len_k)
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
    计算缩放点积注意力，这是现代注意力机制的核心操作

    参数:
        query: 形状为 (batch_size, n_heads, seq_len_q, head_dim)
        key:   形状为 (batch_size, n_heads, seq_len_k, head_dim)
        value: 形状为 (batch_size, n_heads, seq_len_v, head_dim)，通常 seq_len_k = seq_len_v
        mask:  可选的注意力掩码，True 表示需要掩蔽的位置
        dropout_p: 注意力权重的 dropout 概率
        is_causal: 是否使用因果掩码（下三角矩阵）
        scale: 缩放因子，如果为 None，则使用 1/sqrt(head_dim)

    返回:
        注意力加权后的值，形状为 (batch_size, n_heads, seq_len_q, head_dim)
    """
    # 获取维度信息
    head_dim = query.size(-1)

    # 步骤1: 计算注意力分数 (点积)
    attn_scores = attention_score(query, key)

    # 步骤2: 应用缩放因子 - 默认为 1/sqrt(d_k)
    if scale is None:
        scale = math.sqrt(head_dim)
    attn_scores = attn_scores / scale

    # 步骤3: 应用掩码(如果提供)
    if attn_mask is not None:
        attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))

    # 步骤4: 应用因果掩码(如果需要) - 防止信息泄露
    if is_causal:
        seq_len_q, seq_len_k = attn_scores.size(-2), attn_scores.size(-1)
        # 创建下三角矩阵掩码
        causal_mask = torch.triu(torch.ones(seq_len_q, seq_len_k, device=attn_scores.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

    # 步骤5: 应用softmax获取注意力权重
    attn_weights = F.softmax(attn_scores, dim=-1)

    # 步骤6: 应用dropout(如果需要)
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # 步骤7: 用注意力权重加权value
    output = torch.matmul(attn_weights, value)

    return output


# ------------------------------------ 第2部分: 单头注意力 ------------------------------------


class SingleHeadAttention(nn.Module):
    """
    单头自注意力模块 - 帮助理解自注意力机制的最基本形式
    """

    def __init__(self, hidden_size: int, dropout: float = 0.0, is_causal: bool = False):
        """
        参数:
            hidden_size: 输入的隐藏状态维度
            dropout: Dropout概率
            is_causal: 是否使用因果掩码(用于自回归模型)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.is_causal = is_causal

        # 为query、key、value创建线性投影
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)

        # 输出投影
        self.output_proj = nn.Linear(hidden_size, hidden_size)

        # 初始化注意力权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型参数"""
        # 使用正态分布初始化权重
        nn.init.normal_(self.query_proj.weight, std=0.02)
        nn.init.normal_(self.key_proj.weight, std=0.02)
        nn.init.normal_(self.value_proj.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)

        # 将偏置初始化为0
        nn.init.zeros_(self.query_proj.bias)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """
        前向传播

        参数:
            hidden_states: 输入隐藏状态，形状为 (batch_size, seq_len, hidden_size)
            attention_mask: 注意力掩码，形状为 (batch_size, seq_len, seq_len)

        返回:
            注意力层的输出，形状为 (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 步骤1: 生成query, key, value
        query = self.query_proj(hidden_states)  # (batch_size, seq_len, hidden_size)
        key = self.key_proj(hidden_states)  # (batch_size, seq_len, hidden_size)
        value = self.value_proj(hidden_states)  # (batch_size, seq_len, hidden_size)

        # 步骤2: 添加头维度 (做成单头) - (batch_size, 1, seq_len, hidden_size)
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)

        # 步骤3: 处理掩码
        if attention_mask is not None:
            # 扩展掩码维度以便广播
            attention_mask = attention_mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)

        # 步骤4: 计算注意力
        scale_factor = math.sqrt(self.hidden_size)
        attention_output = scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attention_mask,
            dropout_p=self.dropout,
            is_causal=self.is_causal,
            scale=scale_factor,
        )  # (batch_size, 1, seq_len, hidden_size)

        # 步骤5: 移除头维度
        attention_output = attention_output.squeeze(1)  # (batch_size, seq_len, hidden_size)

        # 步骤6: 应用输出投影
        output = self.output_proj(attention_output)  # (batch_size, seq_len, hidden_size)

        return output


# ------------------------------------ 第3部分: 多头注意力 ------------------------------------


class MultiHeadAttention(nn.Module):
    """
    多头注意力模块 - 并行执行多个注意力头，然后将结果合并
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0, is_causal: bool = False):
        """
        参数:
            hidden_size: 输入的隐藏状态维度
            num_heads: 注意力头的数量
            dropout: Dropout概率
            is_causal: 是否使用因果掩码(用于自回归模型)
        """
        super().__init__()
        # 确保hidden_size可以被num_heads整除
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) 必须能被 num_heads ({num_heads}) 整除")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.is_causal = is_causal

        # 每个头的维度
        self.head_dim = hidden_size // num_heads

        # 为query、key、value创建线性投影
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)

        # 输出投影
        self.output_proj = nn.Linear(hidden_size, hidden_size)

        # 初始化注意力权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型参数"""
        # 使用正态分布初始化权重
        nn.init.normal_(self.query_proj.weight, std=0.02)
        nn.init.normal_(self.key_proj.weight, std=0.02)
        nn.init.normal_(self.value_proj.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)

        # 将偏置初始化为0
        nn.init.zeros_(self.query_proj.bias)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.zeros_(self.output_proj.bias)

    def _reshape_for_multihead(self, x: Tensor) -> Tensor:
        """
        将张量重塑为多头格式

        参数:
            x: 形状为 (batch_size, seq_len, hidden_size)

        返回:
            重塑后的张量，形状为 (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, _ = x.shape

        # 重塑为 (batch_size, seq_len, num_heads, head_dim)
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 交换维度，得到 (batch_size, num_heads, seq_len, head_dim)
        return x.transpose(1, 2)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """
        前向传播

        参数:
            hidden_states: 输入隐藏状态，形状为 (batch_size, seq_len, hidden_size)
            attention_mask: 注意力掩码，形状为 (batch_size, seq_len, seq_len)

        返回:
            注意力层的输出，形状为 (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 步骤1: 生成query, key, value投影
        query = self.query_proj(hidden_states)  # (batch_size, seq_len, hidden_size)
        key = self.key_proj(hidden_states)  # (batch_size, seq_len, hidden_size)
        value = self.value_proj(hidden_states)  # (batch_size, seq_len, hidden_size)

        # 步骤2: 将张量重塑为多头格式
        query = self._reshape_for_multihead(query)  # (batch_size, num_heads, seq_len, head_dim)
        key = self._reshape_for_multihead(key)  # (batch_size, num_heads, seq_len, head_dim)
        value = self._reshape_for_multihead(value)  # (batch_size, num_heads, seq_len, head_dim)

        # 步骤3: 处理掩码
        if attention_mask is not None:
            # 扩展掩码维度以便广播
            attention_mask = attention_mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)

        # 步骤4: 计算注意力
        scale_factor = math.sqrt(self.head_dim)  # 注意这里使用head_dim而不是hidden_size
        attention_output = scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attention_mask,
            dropout_p=self.dropout,
            is_causal=self.is_causal,
            scale=scale_factor,
        )  # (batch_size, num_heads, seq_len, head_dim)

        # 步骤5: 重塑回原始形状
        # 首先交换维度 (batch_size, seq_len, num_heads, head_dim)
        attention_output = attention_output.transpose(1, 2)

        # 然后合并多头 (batch_size, seq_len, hidden_size)
        attention_output = attention_output.reshape(batch_size, seq_len, self.hidden_size)

        # 步骤6: 应用输出投影
        output = self.output_proj(attention_output)

        return output


# ------------------------------------ 第4部分: 可视化和演示 ------------------------------------


def visualize_attention(attn_weights: Tensor, input_tokens: list = None, plot_title: str = "Attention Weights") -> None:
    """
    将注意力权重可视化

    参数:
        attn_weights: 注意力权重，形状为 (seq_len_q, seq_len_k)
        input_tokens: 输入序列的标记列表
        plot_title: 图表标题
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # 转换为numpy数组
        attn_weights = attn_weights.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(attn_weights, cmap="viridis")

        # 添加colorbar
        fig.colorbar(im, ax=ax)

        # 设置坐标轴标签
        if input_tokens:
            seq_len = len(input_tokens)
            ax.set_xticks(np.arange(seq_len))
            ax.set_yticks(np.arange(seq_len))
            ax.set_xticklabels(input_tokens)
            ax.set_yticklabels(input_tokens)

            # 让x轴标签倾斜45度
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # 设置标题和坐标轴标签
        ax.set_title(plot_title)
        ax.set_xlabel("Key Pos")
        ax.set_ylabel("Query Pos")

        # 添加数值标注
        for i in range(attn_weights.shape[0]):
            for j in range(attn_weights.shape[1]):
                _text = ax.text(
                    j,
                    i,
                    f"{attn_weights[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="w" if attn_weights[i, j] > 0.5 else "black",
                )

        fig.tight_layout()
        plt.show()
    except ImportError:
        print("请安装matplotlib以便可视化注意力权重。")


def attention_demo():
    """
    一个简单的注意力机制演示
    """
    # 创建一个简单的序列
    batch_size, seq_len, hidden_size, num_heads = 1, 4, 16, 4
    input_seq = torch.randn(batch_size, seq_len, hidden_size)

    print("======== 注意力机制演示 ========")
    print(f"输入形状: {input_seq.shape}")
    print("1. 单头注意力演示")

    # 单头注意力
    single_head_attn = SingleHeadAttention(hidden_size, dropout=0.1)
    single_head_output = single_head_attn(input_seq)
    print(f"单头注意力输出形状: {single_head_output.shape}")

    print("\n2. 多头注意力演示")
    # 多头注意力
    multi_head_attn = MultiHeadAttention(hidden_size, num_heads, dropout=0.1)
    multi_head_output = multi_head_attn(input_seq)
    print(f"多头注意力输出形状: {multi_head_output.shape}")

    print("\n3. 带因果掩码的多头注意力演示")
    multi_head_causal_attn = MultiHeadAttention(hidden_size, num_heads, dropout=0.1, is_causal=True)
    multi_head_causal_output = multi_head_causal_attn(input_seq)
    print(f"因果注意力输出形状: {multi_head_causal_output.shape}")

    # 尝试可视化
    try:
        with torch.no_grad():
            # 获取注意力权重
            query = single_head_attn.query_proj(input_seq).unsqueeze(1)
            key = single_head_attn.key_proj(input_seq).unsqueeze(1)

            # 计算注意力分数
            scores = attention_score(query, key)
            attn_weights = F.softmax(scores / math.sqrt(hidden_size), dim=-1)

            # 可视化注意力权重
            print("\n3. 注意力权重可视化")
            print("请查看生成的图表")
            visualize_attention(
                attn_weights[0, 0],
                input_tokens=[f"Token{i + 1}" for i in range(seq_len)],
                plot_title="Single Head Attn",
            )
    except Exception as e:
        print(f"可视化失败: {e}")


# ------------------------------------ 第5部分: 使用示例 ------------------------------------

if __name__ == "__main__":
    attention_demo()
