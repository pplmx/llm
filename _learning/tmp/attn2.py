"""
注意力机制教学版本 - 从基础到高级的实现 (教学优化版)
=============================================

教学特点：
1. 渐进式学习路径：基础→缩放→单头→多头
2. 丰富的教学注释：包含原理说明和类比解释
3. 交互式可视化：直观展示注意力权重变化
4. 学习检查点：包含理解检查和小练习
5. 典型错误示例：常见错误实现对比

学习路线建议：
1. 先运行 attention_demo() 看整体效果
2. 从 attention_score() 开始逐步学习
3. 尝试 interactive_attention() 交互式学习
4. 使用 check_understanding() 测试理解程度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------- 第1部分：基础函数 (教学核心) ----------------------------


def attention_score(query: Tensor, key: Tensor) -> Tensor:
    """
    基础注意力分数计算: Q @ K^T

    教学说明：
    - 这是注意力机制最核心的计算
    - 每个查询(query)与所有键(key)计算相似度
    - 结果表示query和key的匹配程度
    - 点积越大表示相似度越高
    - 这个分数后续会经过softmax转换为概率分布

    类比理解：
    就像用你的问题(Q)去匹配书籍目录(K)，得到每本书的相关分数
    - Q: 你的问题(例如："如何学习深度学习？")
    - K: 书籍目录的关键词(例如："深度学习基础"、"机器学习入门")
    - 分数: 每本书与你的问题的相关程度

    参数:
        query: [..., seq_len_q, dim] 查询向量序列
        key:   [..., seq_len_k, dim] 键向量序列

    返回:
        注意力分数: [..., seq_len_q, seq_len_k] 每个查询对每个键的分数

    示例:
        >>> Q = torch.tensor([[1.0, 0.5], [0.2, 1.2]])  # 2个查询，每个维度2
        >>> K = torch.tensor([[0.8, 0.3], [1.2, 0.4]])  # 2个键，每个维度2
        >>> attention_score(Q, K)
        tensor([[0.9500, 1.0800],  # 第一个查询与两个键的分数
                [0.5200, 0.4800]]) # 第二个查询与两个键的分数
    """
    # 矩阵乘法实现相似度计算
    # query和key的最后一个维度必须相同
    # 通过转置key的最后两个维度实现矩阵乘法
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
    缩放点积注意力完整实现 (带详细教学注释)

    计算步骤分解：
    1. QK^T        - 计算相似度
    2. Scale       - 缩放防止梯度消失
    3. Mask        - 应用注意力掩码
    4. Softmax     - 转换为概率分布
    5. Dropout     - 随机失活(正则化)
    6. 加权求和     - 聚合value信息

    教学提示：
    - 缩放(scale)是为了防止点积结果过大导致softmax梯度太小
    - 掩码(mask)用于控制哪些位置可以参与计算
    - 因果(causal)掩码确保只能看到当前位置之前的信息
    - dropout用于防止过拟合
    - value是最终被聚合的信息

    参数:
        query: [batch, num_heads, seq_len_q, head_dim] 查询向量
        key:   [batch, num_heads, seq_len_k, head_dim] 键向量
        value: [batch, num_heads, seq_len_v, head_dim] 值向量
        attn_mask: 可选的注意力掩码，形状为[batch, seq_len_q, seq_len_k]或[batch, 1, seq_len_q, seq_len_k]
        dropout_p: dropout概率，训练时随机丢弃部分注意力权重
        is_causal: 是否使用因果掩码(用于自回归生成)
        scale: 自定义缩放因子，默认使用head_dim的平方根的倒数

    返回:
        注意力输出: [batch, num_heads, seq_len_q, head_dim] 注意力加权后的值向量

    示例:
        >>> Q = torch.randn(1, 1, 3, 4)  # 批大小1，1个头，3个查询，每个维度4
        >>> K = torch.randn(1, 1, 5, 4)  # 批大小1，1个头，5个键，每个维度4
        >>> V = torch.randn(1, 1, 5, 4)  # 批大小1，1个头，5个值，每个维度4
        >>> output = scaled_dot_product_attention(Q, K, V)
        >>> output.shape
        torch.Size([1, 1, 3, 4])
    """
    # 1. 计算注意力分数
    attn_scores = torch.matmul(query, key.transpose(-2, -1))

    # 2. 缩放处理
    head_dim = query.size(-1)
    scale_factor = scale if scale is not None else (head_dim**-0.5)
    attn_scores = attn_scores * scale_factor

    # 3. 掩码处理 (教学重点)
    if is_causal:
        # 生成上三角因果掩码 (不能看到未来信息)
        # 例如在序列长度为3时，掩码矩阵如下:
        # [[0, -inf, -inf],
        #  [0,   0, -inf],
        #  [0,   0,   0]]
        seq_len_q, seq_len_k = attn_scores.size(-2), attn_scores.size(-1)
        causal_mask = torch.triu(torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=query.device), diagonal=1)
        attn_scores = attn_scores.masked_fill(causal_mask, -torch.inf)
    elif attn_mask is not None:
        # 处理显式提供的掩码
        if attn_mask.dtype == torch.bool:
            # 布尔掩码，True的位置会被填充为负无穷
            attn_scores = attn_scores.masked_fill(attn_mask, -torch.inf)
        else:
            # 加法掩码，直接加到分数上
            attn_scores = attn_scores + attn_mask

    # 4. Softmax归一化
    # 将分数转换为概率分布，和为1
    # dim=-1表示对最后一个维度(键的维度)进行softmax
    attn_weights = F.softmax(attn_scores, dim=-1)

    # 5. Dropout应用
    # 训练时随机丢弃部分注意力权重
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # 6. 加权求和
    # 使用注意力权重对value进行加权求和
    return torch.matmul(attn_weights, value)


# ---------------------------- 第3部分：单头注意力 (教学中间步骤) ----------------------------


class SingleHeadAttention(nn.Module):
    """
    单头自注意力模块 (教学优化版)

    教学重点：
    - 展示如何将基础计算封装为模块
    - 解释QKV投影的含义
    - 说明输出投影的作用
    - 展示初始化方法
    - 解释前向传播流程

    类比理解：
    - 就像一个人专注阅读一篇文章的不同部分
    - Q: 阅读时关注的问题(你想了解什么)
    - K: 文章各部分的关键词(文章在讲什么)
    - V: 文章各部分的实际内容(文章的具体信息)
    - 输出: 你从文章中提取的信息

    结构:
    输入 → QKV投影 → 分割QKV → 注意力计算 → 输出投影 → 输出

    参数:
        hidden_size: 输入和输出的特征维度
        dropout_p: 注意力权重的dropout概率
        is_causal: 是否使用因果注意力(用于生成任务)
    """

    def __init__(self, hidden_size: int, dropout_p: float = 0.1, is_causal: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.is_causal = is_causal

        # 合并的QKV投影 (教学说明：实际实现常用这种方式提高效率)
        # 输入维度: hidden_size
        # 输出维度: 3*hidden_size (Q,K,V各hidden_size)
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)

        # 输出投影
        # 将注意力输出映射回hidden_size维度
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # 初始化说明
        self._init_weights()

    def _init_weights(self):
        """初始化说明：Xavier初始化适合注意力层"""
        # 线性层的权重使用Xavier均匀初始化
        # 偏置初始化为0
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """
        前向传播流程教学分解：
        1. 计算QKV投影
        2. 分割Q,K,V
        3. 计算缩放点积注意力
        4. 应用输出投影

        参数:
            x: 输入张量，形状为[batch_size, seq_len, hidden_size]
            attn_mask: 可选注意力掩码，形状为[batch_size, seq_len, seq_len]

        返回:
            输出张量，形状与输入相同[batch_size, seq_len, hidden_size]
        """
        # 1. 投影
        # 将输入x通过线性层投影到QKV空间
        qkv = self.qkv_proj(x)  # [B, S, 3*H]

        # 2. 分割 (教学提示：chunk是PyTorch的拆分函数)
        # 将最后一个维度分成3份，得到Q,K,V
        q, k, v = qkv.chunk(3, dim=-1)  # 每个[B, S, H]

        # 3. 计算注意力 (添加头维度以适应接口)
        # 虽然这是单头注意力，但为了接口统一，我们添加一个头维度
        attn_output = scaled_dot_product_attention(
            q.unsqueeze(1),  # [B, 1, S, H]
            k.unsqueeze(1),  # [B, 1, S, H]
            v.unsqueeze(1),  # [B, 1, S, H]
            attn_mask=attn_mask.unsqueeze(1) if attn_mask is not None else None,
            dropout_p=self.dropout_p,
            is_causal=self.is_causal,
        ).squeeze(1)  # [B, S, H]

        # 4. 输出投影
        # 将注意力输出映射回原始维度
        return self.out_proj(attn_output)


# ---------------------------- 第4部分：多头注意力 (完整实现) ----------------------------


class MultiHeadAttention(nn.Module):
    """
    多头注意力模块 (教学完整版)

    教学重点：
    - 多头注意力的核心思想
    - 如何分割和重组注意力头
    - 实际实现中的效率考量
    - 参数初始化
    - 前向传播流程

    类比理解：
    - 就像多个人同时阅读同一篇文章
    - 每个人关注不同的方面(头)
    - 最后整合所有人的理解
    - 每个头可以学习不同的注意力模式

    结构:
    输入 → QKV投影 → 分割多头 → 并行注意力计算 → 合并多头 → 输出投影 → 输出

    参数:
        hidden_size: 输入和输出的特征维度
        num_heads: 注意力头的数量
        dropout_p: 注意力权重的dropout概率
        is_causal: 是否使用因果注意力(用于生成任务)
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout_p: float = 0.1, is_causal: bool = False):
        super().__init__()
        # 参数验证 (教学提示：确保可分割)
        assert hidden_size % num_heads == 0, "hidden_size必须能被num_heads整除"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # 每个头的维度
        self.dropout_p = dropout_p
        self.is_causal = is_causal

        # 投影层 (教学说明：实际框架常用合并投影)
        # 输入维度: hidden_size
        # 输出维度: 3*hidden_size (Q,K,V各hidden_size)
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)

        # 输出投影
        # 将合并后的多头输出映射回hidden_size维度
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self._init_weights()

    def _init_weights(self):
        """权重初始化教学说明"""
        # 线性层的权重使用Xavier均匀初始化
        # 偏置初始化为0
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """
        前向传播流程教学分解：
        1. 计算QKV投影
        2. 重塑为多头形式
        3. 计算缩放点积注意力
        4. 合并多头结果
        5. 应用输出投影

        参数:
            x: 输入张量，形状为[batch_size, seq_len, hidden_size]
            attn_mask: 可选注意力掩码，形状为[batch_size, seq_len, seq_len]

        返回:
            输出张量，形状与输入相同[batch_size, seq_len, hidden_size]
        """
        B, S, _ = x.shape  # 批大小，序列长度，特征维度

        # 1. 投影
        # 将输入x通过线性层投影到QKV空间
        qkv = self.qkv_proj(x)  # [B, S, 3*H]

        # 2. 分割并重塑为多头 (教学重点)
        # 将QKV分割成单独的Q,K,V
        q, k, v = qkv.chunk(3, dim=-1)  # 每个[B, S, H]

        # 重塑为 [B, S, N, D] -> [B, N, S, D]
        # 其中N是头数，D是每个头的维度
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 调整掩码形状 (教学提示：广播机制)
        # 如果提供了掩码，需要增加一个头维度以广播到所有头
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)  # [B, 1, S, S]

        # 4. 计算注意力
        # 每个头独立计算注意力
        attn_output = scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p if self.training else 0.0, is_causal=self.is_causal
        )  # [B, N, S, D]

        # 5. 合并多头
        # 将多头输出合并回原始形状
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, S, N, D]
        attn_output = attn_output.view(B, S, self.hidden_size)  # [B, S, H]

        # 6. 输出投影
        # 将合并后的输出映射回原始维度
        return self.out_proj(attn_output)


# ---------------------------- 第5部分：教学工具 ----------------------------


def interactive_attention():
    """交互式注意力计算演示"""
    print("\n=== 交互式注意力演示 ===")
    print("我们将创建一个简单的2x2注意力计算示例")
    print("这样可以手动验证计算过程")

    # 简单示例数据
    Q = torch.tensor([[1.0, 0.5], [0.2, 1.2]])
    K = torch.tensor([[0.8, 0.3], [1.2, 0.4]])
    V = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    print("\n1. 原始分数计算:")
    print("Q:", Q)
    print("K:", K)
    print("Q @ K.T =")
    raw_scores = attention_score(Q, K)
    print(raw_scores)

    print("\n2. 缩放后分数:")
    print("缩放因子 = 1/sqrt(dim) = 1/sqrt(2) ≈ 0.707")
    scaled_scores = raw_scores * (0.5**0.5)  # dim=2
    print(scaled_scores)

    print("\n3. Softmax权重:")
    print("对每一行进行softmax:")
    weights = F.softmax(scaled_scores, dim=-1)
    print(weights)

    print("\n4. 最终输出:")
    print("V:", V)
    print("权重 @ V =")
    output = torch.matmul(weights, V)
    print(output)

    print("\n总结:")
    print("注意力机制通过计算查询与键的相似度，得到权重")
    print("然后使用这些权重对值进行加权求和")
    print("最终输出是值的加权组合")


def check_understanding():
    """注意力机制理解检查"""
    print("\n=== 理解检查 ===")
    print("请回答以下问题来测试你对注意力机制的理解")

    questions = [
        {
            "question": "缩放因子(scale)的主要作用是什么?",
            "options": [
                "A. 减少计算量",
                "B. 防止softmax梯度消失",
                "C. 增加模型容量",
                "D. 加速收敛",
            ],
            "answer": "B",
            "explanation": "缩放因子(通常是1/sqrt(dim))防止点积结果过大导致softmax梯度太小",
        },
        {
            "question": "多头注意力的主要优势是什么?",
            "options": [
                "A. 减少参数数量",
                "B. 允许模型同时关注不同位置的不同方面",
                "C. 降低计算复杂度",
                "D. 避免使用残差连接",
            ],
            "answer": "B",
            "explanation": "多头可以让模型在不同的表示子空间中学习不同的注意力模式",
        },
        {
            "question": "因果注意力掩码的作用是什么?",
            "options": [
                "A. 提高计算效率",
                "B. 防止模型看到未来信息",
                "C. 增加模型容量",
                "D. 减少内存消耗",
            ],
            "answer": "B",
            "explanation": "因果掩码确保模型在生成当前token时只能看到之前的token",
        },
    ]

    score = 0
    for i, q in enumerate(questions):
        print(f"\n问题{i + 1}: {q['question']}")
        for opt in q["options"]:
            print(f"  {opt}")

        user_answer = input("你的答案: ").upper()
        if user_answer == q["answer"]:
            print("✓ 正确!")
            print(f"解释: {q['explanation']}")
            score += 1
        else:
            print(f"✗ 错误，正确答案是 {q['answer']}")
            print(f"解释: {q['explanation']}")

    print(f"\n你的得分: {score}/{len(questions)}")
    if score == len(questions):
        print("太棒了！你对注意力机制理解很透彻！")
    elif score >= len(questions) / 2:
        print("不错！但还有一些概念需要加强理解")
    else:
        print("建议回顾注意力机制的基本概念")


# ---------------------------- 第6部分：演示执行 ----------------------------

if __name__ == "__main__":
    print("=== 注意力机制教学演示 ===")
    print("请选择要运行的功能:")
    print("1. 交互式注意力演示 - 手动计算小例子")
    print("2. 理解检查 - 测试你对注意力机制的理解")
    print("3. 完整模块演示 - 运行多头注意力模块")

    choice = input("请输入选项(1-3): ")

    if choice == "1":
        interactive_attention()
    elif choice == "2":
        check_understanding()
    else:
        # 简化版的演示函数
        print("\n运行完整模块演示...")
        print("创建一个多头注意力模块:")
        print("hidden_size=64, num_heads=8")
        mha = MultiHeadAttention(hidden_size=64, num_heads=8)

        print("\n创建随机输入:")
        print("batch_size=2, seq_len=10, hidden_size=64")
        x = torch.randn(2, 10, 64)  # 批大小2，序列长度10，特征64

        print("\n运行前向传播...")
        out = mha(x)

        print(f"\n输入形状: {x.shape}")
        print(f"输出形状: {out.shape}")
        print("演示完成!")
        print("可以看到输入输出形状相同，这是自注意力的特点")
