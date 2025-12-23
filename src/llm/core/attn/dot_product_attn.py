import math

import torch
import torch.nn.functional as F
from torch import Tensor


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
    计算缩放点积注意力 (Scaled Dot-Product Attention).

    参数:
        query (Tensor): 查询张量, 形状 (B, N, Sq, D). B=批次, N=头数, Sq=查询序列长度, D=头维度.
        key (Tensor): 键张量, 形状 (B, N, Sk, D). Sk=键序列长度.
        value (Tensor): 值张量, 形状 (B, N, Sv, D). 通常 Sk == Sv.
        attn_mask (Tensor | None): 布尔注意力掩码.
            - 形状需能广播至 (B, N, Sq, Sk).
            - `True` 表示屏蔽该位置 (将其注意力分数设为 -inf).
            - 默认为 None.
        dropout_p (float): 应用于注意力权重的 dropout 概率. 仅在 > 0 时应用. 默认为 0.0.
        is_causal (bool): 若为 True, 应用因果掩码 (上三角掩码), 阻止关注未来位置.
                          会与 attn_mask 结合使用 (逻辑或). 默认为 False.
        scale (float | None): 缩放因子. 若为 None, 则使用 1 / sqrt(head_dim). 默认为 None.

    返回:
        Tensor: 注意力输出张量, 形状 (B, N, Sq, D).
    """
    head_dim = query.size(-1)
    # 若未提供 scale, 则使用 1/sqrt(d_k) 作为默认值 (d_k == head_dim)
    scale = scale or 1.0 / math.sqrt(head_dim)

    # 2. 计算注意力分数并应用缩放: (Q @ K^T) * scale
    # (B, N, Sq, D) @ (B, N, D, Sk) -> (B, N, Sq, Sk)
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    # 3. 应用掩码 (组合 is_causal 和 attn_mask)
    mask_value = -torch.inf  # 或者使用 torch.finfo(attn_scores.dtype).min
    final_mask = None

    if is_causal:
        # 创建因果掩码 (上三角矩阵, 对角线为 False)
        # True 表示需要屏蔽的位置 (未来位置)
        seq_len_q, seq_len_k = attn_scores.shape[-2:]
        # causal_mask 的形状是 (Sq, Sk) 或 (1, 1, Sq, Sk) 以便广播
        causal_mask = torch.triu(
            torch.ones((seq_len_q, seq_len_k), device=query.device, dtype=torch.bool),
            diagonal=1,  # 屏蔽严格上三角部分 (i > j)
        )
        final_mask = causal_mask  # .unsqueeze(0).unsqueeze(0) # 根据需要调整维度以广播, 但通常 PyTorch 会自动处理

    if attn_mask is not None:
        if attn_mask.dtype != torch.bool:
            raise ValueError(f"attn_mask 必须是布尔类型张量, 但收到 {attn_mask.dtype}")
        # 检查 attn_mask 的维度是否可以广播到 attn_scores
        # expected_shape = (..., seq_len_q, seq_len_k)
        # (这里省略显式检查以保持简洁, 依赖 PyTorch 的广播机制)

        final_mask = attn_mask if final_mask is None else final_mask | attn_mask

    # 应用最终计算出的掩码
    if final_mask is not None:
        # 使用 mask_fill 将需要屏蔽的位置 (True) 设置为极小值
        # PyTorch 的广播机制会自动处理 final_mask 和 attn_scores 之间的维度匹配
        attn_scores = attn_scores.masked_fill(final_mask, mask_value)

    # 4. 计算注意力权重: softmax(scores)
    # 在最后一个维度 (key 序列长度维度 Sk) 上进行 softmax
    attn_weights = F.softmax(attn_scores, dim=-1)

    # 5. 应用 Dropout (仅在 dropout_p > 0 时)
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)  # 训练/评估模式由外部模块控制

    # 6. 计算加权值: weights @ V
    # (B, N, Sq, Sk) @ (B, N, Sv, D) -> (B, N, Sq, D)  (假定 Sk == Sv)
    output = torch.matmul(attn_weights, value)

    return output
