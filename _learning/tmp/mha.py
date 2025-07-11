import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制实现。

    集成了Layer Normalization和残差连接，支持Pre-LN和Post-LN模式。
    提供了高效的计算实现和灵活的配置选项。

    Args:
        hidden_size (int): 模型的总维度。
        num_heads (int): 注意力头的数量，必须能整除hidden_size。默认为8。
        p (float): 应用于注意力权重和最终输出的dropout概率。默认为0.1。
        bias (bool): 是否在线性层（QKV投影和输出投影）中使用偏置。默认为False。
        eps (float): Layer Normalization的epsilon值。默认为1e-5。
        norm_first (bool): 是否使用Pre-LN（True）或Post-LN（False）架构。默认为True。
        is_causal (bool): 是否默认应用因果掩码（如用于解码器）。默认为False。
        separate_qkv (bool): 是否使用分离的Q、K、V投影层。默认为False（使用融合QKV投影）。
        device (torch.device | str | None): 模型参数的目标设备。默认为None（自动推断）。
        dtype (torch.dtype | None): 模型参数的目标数据类型。默认为None（自动推断）。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        p: float = 0.1,
        bias: bool = False,
        eps: float = 1e-5,
        norm_first: bool = True,
        is_causal: bool = False,
        separate_qkv: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")

        factory_kwargs = {"device": device, "dtype": dtype}
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.norm_first = norm_first
        self.is_causal = is_causal
        self.separate_qkv = separate_qkv
        self.p = p
        self.dropout = nn.Dropout(p)

        # --- Layers ---
        self.norm = nn.LayerNorm(hidden_size, eps=eps, **factory_kwargs)

        # 选择使用融合QKV还是分离QKV投影
        if separate_qkv:
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias, **factory_kwargs)
        else:
            self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=bias, **factory_kwargs)

        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias, **factory_kwargs)

        self._init_weights()

    def _init_weights(self):
        """初始化线性层权重（Xavier均匀分布）和偏置（零）。"""
        if self.separate_qkv:
            for proj in [self.q_proj, self.k_proj, self.v_proj]:
                nn.init.xavier_uniform_(proj.weight)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)
        else:
            nn.init.xavier_uniform_(self.qkv_proj.weight)
            if self.qkv_proj.bias is not None:
                nn.init.zeros_(self.qkv_proj.bias)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _reshape_for_attention(self, x: Tensor, batch_size: int, seq_len: int) -> Tensor:
        """将投影后的张量重新整形为注意力计算所需的形状。"""
        # [B, S, H] -> [B, S, N, D] -> [B, N, S, D]
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool | None = None,
    ) -> Tensor:
        """
        前向传播。

        Args:
            hidden_states (Tensor): 输入张量，形状为 [B, S, H]（批量大小，序列长度，隐藏大小）。
            attn_mask (Tensor | None): 可选的注意力掩码。
                - 对于F.scaled_dot_product_attention，期望是一个布尔张量，其中`True`表示掩蔽。
                - 形状应可广播到 [B, N, S, S]（批量大小，头数，序列长度，序列长度）。
                - 例如，填充掩码可以是 [B, 1, 1, S] 或 [B, 1, S, S]。
            is_causal (bool | None): 是否在此次前向传播中强制执行因果掩码。
                - 如果为`None`（默认），使用初始化期间设置的默认值`self.is_causal`。
                - 如果为`True`或`False`，将覆盖默认设置。

        Returns:
            Tensor: 输出张量，形状为 [B, S, H]。
        """
        batch_size, seq_len, _ = hidden_states.size()
        residual = hidden_states

        # --- 确定此次调用的因果设置 ---
        use_causal = self.is_causal if is_causal is None else is_causal

        # --- 1. Layer Normalization (Pre-LN模式) ---
        if self.norm_first:
            hidden_states = self.norm(hidden_states)

        # --- 2. 投影Q、K、V并重新整形 ---
        if self.separate_qkv:
            # 使用分离的投影层
            q = self._reshape_for_attention(self.q_proj(hidden_states), batch_size, seq_len)
            k = self._reshape_for_attention(self.k_proj(hidden_states), batch_size, seq_len)
            v = self._reshape_for_attention(self.v_proj(hidden_states), batch_size, seq_len)
        else:
            # 使用融合的QKV投影（更高效）
            qkv = self.qkv_proj(hidden_states)  # [B, S, 3*H]
            qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
            q, k, v = [qkv[:, :, i].transpose(1, 2) for i in range(3)]  # 每个 [B, N, S, D]

        # --- 3. 注意力计算 ---
        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.p if self.training else 0.0,
            is_causal=use_causal,
        )  # 输出形状: [B, N, S, D]

        # --- 4. 合并头部输出 ---
        # [B, N, S, D] -> [B, S, N, D] -> [B, S, H]
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)

        # --- 5. 输出投影和dropout ---
        output = self.dropout(self.out_proj(attn_output))

        # --- 6. 残差连接 ---
        output = output + residual

        # --- 7. Layer Normalization (Post-LN模式) ---
        if not self.norm_first:
            output = self.norm(output)

        return output
