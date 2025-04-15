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
    attn_mask: Tensor | None = None,  # attn_mask 可以是 None
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,  # scale 可以是 None
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
    effective_scale = scale if scale is not None else (head_dim**-0.5)

    # 2. 计算注意力分数并应用缩放: (Q @ K^T) * scale
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * effective_scale

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


# ------------------------------------ 第2部分: 单头注意力 (优化版) ------------------------------------


class SingleHeadAttention(nn.Module):
    """
    单头自注意力模块 (优化: 使用合并的 QKV 投影)。
    """

    def __init__(self, hidden_size: int, p: float = 0.0, is_causal: bool = False, bias: bool = True):
        """
        参数:
            hidden_size (int): 输入/输出的隐藏状态维度。
            p (float): 注意力权重的 Dropout 概率。
            is_causal (bool): 是否使用因果掩码。
            bias (bool): 线性层是否使用偏置。
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.p = p  # 注意力内部 dropout 概率
        self.is_causal = is_causal

        # 合并的 Q, K, V 线性投影层
        self.in_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        # 输出投影层
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        # 输出 dropout 层 (可选，这里用注意力 p，也可单独设置)
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
            attn_mask (Optional[Tensor]): 注意力掩码 [B, S, S] 或 [B, 1, S, S]。

        返回:
            Tensor: 注意力层输出 [B, S, H]。
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 1. 计算 Q, K, V 投影并分割
        qkv = self.in_proj(hidden_states)  # [B, S, 3*H]
        query, key, value = qkv.chunk(3, dim=-1)  # 每个都是 [B, S, H]

        # 2. 添加 "头" 维度 (N=1) -> [B, 1, S, H]
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)

        # 3. 调整掩码维度 (如果需要)
        if attn_mask is not None:
            # 确保 mask 形状是 [B, 1, Sq, Sk] 或可广播
            if attn_mask.dim() == 3:  # e.g., [B, S, S] -> [B, 1, S, S]
                attn_mask = attn_mask.unsqueeze(1)

        # 4. 计算缩放点积注意力
        # 对于单头，头维度 D == H
        attn_output = scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            # 只有在训练时才应用 dropout
            dropout_p=self.p if self.training else 0.0,
            is_causal=self.is_causal,
            scale=None,  # 让函数内部计算 scale = 1/sqrt(H)
        )  # 输出 [B, 1, S, H]

        # 5. 移除 "头" 维度 -> [B, S, H]
        attn_output = attn_output.squeeze(1)

        # 6. 应用输出投影和 dropout
        output = self.dropout(self.out_proj(attn_output))  # [B, S, H]

        return output


# ------------------------------------ 第3部分: 多头注意力 (优化版) ------------------------------------


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


# ------------------------------------ 第4部分: 可视化和演示 ------------------------------------


def visualize_attention(
    attn_weights: Tensor, input_tokens: list | None = None, plot_title: str = "Attention Weights"
):
    """
    可视化注意力权重。

    参数:
        attn_weights (Tensor): 注意力权重, 形状 (seq_len_q, seq_len_k)。
        input_tokens (Optional[list]): 输入序列的标记列表 (用于坐标轴标签)。
        plot_title (str): 图表标题。
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置中文字体，例如 SimHei
        plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

        # 将张量移至 CPU 并转为 NumPy 数组
        attn_weights = attn_weights.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 6))  # 调整图像大小
        im = ax.imshow(attn_weights, cmap="viridis")  # 使用 viridis 色彩映射

        # 添加颜色条
        fig.colorbar(im, ax=ax)

        # 设置坐标轴标签
        if input_tokens:
            seq_len_q, seq_len_k = attn_weights.shape
            ax.set_xticks(np.arange(seq_len_k))
            ax.set_yticks(np.arange(seq_len_q))
            ax.set_xticklabels(input_tokens[:seq_len_k])
            ax.set_yticklabels(input_tokens[:seq_len_q])

            # 旋转 X 轴标签防止重叠
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        else:
            ax.set_xlabel("键 (Key) 位置")
            ax.set_ylabel("查询 (Query) 位置")

        # 设置标题
        ax.set_title(plot_title)

        # 在每个单元格显示权重数值
        thresh = attn_weights.max() / 2.0  # 阈值用于决定文字颜色
        for i in range(attn_weights.shape[0]):
            for j in range(attn_weights.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{attn_weights[i, j]:.2f}",  # 格式化为两位小数
                    ha="center",
                    va="center",
                    color="white" if attn_weights[i, j] > thresh else "black",
                )

        fig.tight_layout()  # 自动调整布局
        plt.show()
    except ImportError:
        print("请安装 matplotlib 和 numpy 以进行可视化: pip install matplotlib numpy")
    except Exception as e:
        print(f"可视化出错: {e}")


def attention_demo():
    """
    注意力机制的简单演示。
    """
    # --- 参数设置 ---
    batch_size, seq_len, hidden_size, num_heads = 1, 5, 32, 4  # 增加序列长度和维度
    dropout_p = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择设备

    # 创建输入序列
    input_seq = torch.randn(batch_size, seq_len, hidden_size, device=device)
    # 创建一个示例掩码 (例如，屏蔽最后一个 token 的 key)
    # 注意: mask 通常表示要 *屏蔽* 的位置 (True 或 -inf)
    example_mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool, device=device)
    example_mask[:, :, -1] = True  # 屏蔽所有 query 对最后一个 key 的注意力

    print("======== 注意力机制演示 ========")
    print(f"设备: {device}")
    print(f"输入形状: {input_seq.shape}")
    print(f"头数: {num_heads}, 头维度: {hidden_size // num_heads}")

    # --- 1. 单头注意力 ---
    print("\n--- 1. 单头注意力 (合并投影) ---")
    single_head_attn = SingleHeadAttention(hidden_size, p=dropout_p, is_causal=False).to(device)
    single_head_attn.eval()  # 设置为评估模式以禁用 dropout (用于权重提取)
    with torch.no_grad():  # 无需计算梯度
        single_head_output = single_head_attn(input_seq)
    print(f"单头注意力输出形状: {single_head_output.shape}")

    # --- 2. 多头注意力 ---
    print("\n--- 2. 多头注意力 (合并投影) ---")
    multi_head_attn = MultiHeadAttention(hidden_size, num_heads, p=dropout_p, is_causal=False).to(device)
    multi_head_attn.eval()  # 评估模式
    with torch.no_grad():
        multi_head_output = multi_head_attn(input_seq)
        # 应用掩码的例子
        multi_head_output_masked = multi_head_attn(input_seq, attn_mask=example_mask)
    print(f"多头注意力输出形状: {multi_head_output.shape}")
    print(f"带掩码的多头注意力输出形状: {multi_head_output_masked.shape}")

    # --- 3. 因果多头注意力 ---
    print("\n--- 3. 因果多头注意力 ---")
    multi_head_causal_attn = MultiHeadAttention(hidden_size, num_heads, p=dropout_p, is_causal=True).to(device)
    multi_head_causal_attn.eval()  # 评估模式
    with torch.no_grad():
        multi_head_causal_output = multi_head_causal_attn(input_seq)
    print(f"因果多头注意力输出形状: {multi_head_causal_output.shape}")

    # --- 4. 注意力权重可视化 ---
    print("\n--- 4. 注意力权重可视化 (多头中的第一个头) ---")
    print("请查看弹出的图表。")
    tokens = [f"词{i + 1}" for i in range(seq_len)]  # 示例词元

    # 提取多头注意力的权重 (第一个样本，第一个头)
    try:
        multi_head_attn.eval()  # 确保是评估模式
        with torch.no_grad():
            # 1. 获取 Q, K
            qkv = multi_head_attn.in_proj(input_seq)
            query, key, _ = qkv.chunk(3, dim=-1)
            # 2. 重塑 Q, K
            query = query.view(batch_size, seq_len, num_heads, multi_head_attn.head_dim).transpose(1, 2)  # [B,N,S,D]
            key = key.view(batch_size, seq_len, num_heads, multi_head_attn.head_dim).transpose(1, 2)  # [B,N,S,D]
            # 3. 计算缩放分数 (不加掩码，不加 dropout)
            scale = multi_head_attn.head_dim**-0.5
            scores = torch.matmul(query, key.transpose(-2, -1)) * scale  # [B,N,S,S]
            # 4. Softmax
            attn_weights = F.softmax(scores, dim=-1)  # [B,N,S,S]

        # 可视化第一个样本的第一个头的权重
        visualize_attention(
            attn_weights[0, 0],  # 选择 [样本0, 头0]
            input_tokens=tokens,
            plot_title=f"多头注意力权重 (头 1/{num_heads})",
        )

        # 可视化带因果掩码的权重
        multi_head_causal_attn.eval()
        with torch.no_grad():
            qkv_causal = multi_head_causal_attn.in_proj(input_seq)
            query_c, key_c, _ = qkv_causal.chunk(3, dim=-1)
            query_c = query_c.view(batch_size, seq_len, num_heads, multi_head_causal_attn.head_dim).transpose(1, 2)
            key_c = key_c.view(batch_size, seq_len, num_heads, multi_head_causal_attn.head_dim).transpose(1, 2)
            scale_c = multi_head_causal_attn.head_dim**-0.5
            scores_c = torch.matmul(query_c, key_c.transpose(-2, -1)) * scale_c
            # 手动应用因果掩码以便可视化
            causal_mask_vis = torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)
            scores_c = scores_c.masked_fill(causal_mask_vis, -torch.inf)
            attn_weights_causal = F.softmax(scores_c, dim=-1)

        visualize_attention(
            attn_weights_causal[0, 0], input_tokens=tokens, plot_title=f"因果多头注意力权重 (头 1/{num_heads})"
        )

    except Exception as e:

        print(f"\n可视化失败: {e}")
        # traceback.print_exc() # 如果需要详细错误栈


# ------------------------------------ 第5部分: 执行入口 ------------------------------------

if __name__ == "__main__":
    # 确保 matplotlib 支持中文显示 (如果需要)
    # 可以取消下面这行的注释来尝试设置字体，或者在 visualize_attention 中设置
    # plt.rcParams['font.sans-serif'] = ['SimHei'] # 或者其他你系统支持的中文字体
    # plt.rcParams['axes.unicode_minus'] = False

    attention_demo()
