from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class MultiHeadAttentionConfig:
    """多头注意力的配置类。"""

    hidden_size: int
    num_heads: int = 8
    dropout: float = 0.1
    attention_dropout: float | None = None  # 如果未指定，则使用dropout值
    bias: bool = False
    eps: float = 1e-5
    norm_first: bool = True
    is_causal: bool = False
    separate_qkv: bool = False
    use_rotary_embeddings: bool = False
    rotary_dim: int | None = None  # 若使用旋转位置嵌入，可指定维度
    kv_cache_enabled: bool = False
    device: torch.device | str | None = None
    dtype: torch.dtype | None = None

    def __post_init__(self):
        # 验证配置
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})")

        # 设置默认值
        if self.attention_dropout is None:
            self.attention_dropout = self.dropout

        if self.use_rotary_embeddings and self.rotary_dim is None:
            # 默认使用全部维度
            self.rotary_dim = self.hidden_size // self.num_heads


class EnhancedMultiHeadAttention(nn.Module):
    """
    增强的多头注意力实现，使用配置类进行初始化。

    支持多种高级功能:
    - 旋转位置嵌入 (RoPE)
    - KV缓存
    - 分离或融合的QKV投影
    - 可配置的归一化策略
    """

    def __init__(self, config: MultiHeadAttentionConfig):
        super().__init__()
        self.config = config

        factory_kwargs = {"device": config.device, "dtype": config.dtype}
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.norm_first = config.norm_first
        self.is_causal = config.is_causal
        self.p = config.dropout
        self.attn_p = config.attention_dropout
        self.scale = 1.0 / (self.head_dim**0.5)

        # --- 层 ---
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.eps, **factory_kwargs)

        # QKV投影策略
        if config.separate_qkv:
            self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias, **factory_kwargs)
            self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias, **factory_kwargs)
            self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias, **factory_kwargs)
            self.has_separate_qkv = True
        else:
            self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.bias, **factory_kwargs)
            self.has_separate_qkv = False

        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias, **factory_kwargs)
        self.dropout = nn.Dropout(config.dropout)

        # 如果使用旋转位置嵌入
        self.use_rope = config.use_rotary_embeddings
        if self.use_rope:
            self.rotary_dim = config.rotary_dim

        self._init_weights()

    def _init_weights(self):
        """初始化线性层权重（Xavier均匀分布）和偏置（零）。"""
        if self.has_separate_qkv:
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

    def _apply_rotary_pos_emb(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
        """应用旋转位置嵌入 (RoPE)。"""
        # 实现RoPE操作
        # 仅应用于前rotary_dim维度
        q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
        k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]

        # 应用旋转
        q_rot_cos = cos.unsqueeze(1) * q_rot  # [B, 1, S, D_r]
        q_rot_sin = sin.unsqueeze(1) * q_rot.roll(shifts=1, dims=-1)
        k_rot_cos = cos.unsqueeze(1) * k_rot
        k_rot_sin = sin.unsqueeze(1) * k_rot.roll(shifts=1, dims=-1)

        # 合并
        q_out = torch.cat([q_rot_cos - q_rot_sin, q_pass], dim=-1)
        k_out = torch.cat([k_rot_cos - k_rot_sin, k_pass], dim=-1)

        return q_out, k_out

    def _reshape_for_attention(self, x: Tensor, batch_size: int, seq_len: int) -> Tensor:
        """将投影后的张量重新整形为注意力计算所需的形状。"""
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool | None = None,
        position_ids: Tensor | None = None,
        past_key_value: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        前向传播。

        Args:
            hidden_states (Tensor): 输入张量，形状为 [B, S, H]
            attn_mask (Tensor, optional): 注意力掩码
            is_causal (bool, optional): 是否使用因果掩码，覆盖默认设置
            position_ids (Tensor, optional): 位置ID，用于旋转位置嵌入
            past_key_value (Tuple[Tensor, Tensor], optional): 之前的KV缓存
            use_cache (bool): 是否返回更新的KV缓存

        Returns:
            输出张量，以及可选的KV缓存（如果use_cache=True）
        """
        batch_size, seq_len, _ = hidden_states.size()
        residual = hidden_states

        # --- 确定此次调用的因果设置 ---
        use_causal = self.is_causal if is_causal is None else is_causal

        # --- 1. Layer Normalization (Pre-LN模式) ---
        if self.norm_first:
            hidden_states = self.norm(hidden_states)

        # --- 2. 投影Q、K、V并重新整形 ---
        if self.has_separate_qkv:
            # 使用分离的投影层
            q = self.q_proj(hidden_states)  # [B, S, H]

            # 处理KV缓存
            if past_key_value is not None and self.config.kv_cache_enabled:
                # 使用缓存时，只计算最后一个token的KV
                k = self.k_proj(hidden_states[:, -1:, :])  # [B, 1, H]
                v = self.v_proj(hidden_states[:, -1:, :])  # [B, 1, H]

                # 与缓存连接
                k = torch.cat([past_key_value[0], k], dim=1)  # [B, S+1, H]
                v = torch.cat([past_key_value[1], v], dim=1)  # [B, S+1, H]
            else:
                # 首次计算，处理整个序列
                k = self.k_proj(hidden_states)  # [B, S, H]
                v = self.v_proj(hidden_states)  # [B, S, H]
        else:
            # 使用融合的QKV投影
            if past_key_value is not None and self.config.kv_cache_enabled:
                # 对最后一个token进行QKV投影
                q = self._reshape_for_attention(
                    self.qkv_proj(hidden_states)[:, :, : self.hidden_size], batch_size, seq_len
                )  # [B, N, S, D]

                # 只对新token计算K和V
                new_k = self._reshape_for_attention(
                    self.qkv_proj(hidden_states[:, -1:, :])[:, :, self.hidden_size : 2 * self.hidden_size],
                    batch_size,
                    1,
                )  # [B, N, 1, D]

                new_v = self._reshape_for_attention(
                    self.qkv_proj(hidden_states[:, -1:, :])[:, :, 2 * self.hidden_size :], batch_size, 1
                )  # [B, N, 1, D]

                # 与缓存连接
                k = torch.cat([past_key_value[0], new_k], dim=2)  # [B, N, S+1, D]
                v = torch.cat([past_key_value[1], new_v], dim=2)  # [B, N, S+1, D]
            else:
                # 常规QKV投影
                qkv = self.qkv_proj(hidden_states)  # [B, S, 3*H]
                qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
                q, k, v = [qkv[:, :, i].transpose(1, 2) for i in range(3)]  # 每个 [B, N, S, D]

        # 确保所有张量都是正确的形状
        if not self.has_separate_qkv or past_key_value is None or not self.config.kv_cache_enabled:
            q = self._reshape_for_attention(q, batch_size, seq_len) if q.dim() == 3 else q
            k = self._reshape_for_attention(k, batch_size, k.size(1)) if k.dim() == 3 else k
            v = self._reshape_for_attention(v, batch_size, v.size(1)) if v.dim() == 3 else v

        # 应用旋转位置嵌入（如果启用）
        if self.use_rope and position_ids is not None:
            # 计算正弦和余弦
            seq_length = position_ids.size(1)
            position_ids = position_ids.view(-1, seq_length)

            # 生成角度，考虑到旋转位置嵌入的维度
            inv_freq = 1.0 / (
                10000 ** (torch.arange(0, self.rotary_dim, 2, device=position_ids.device) / self.rotary_dim)
            )
            sincos = torch.einsum("i,j->ij", position_ids.flatten(), inv_freq)
            sin, cos = sincos.sin(), sincos.cos()

            # 重塑成正确的形状
            sin = sin.view(batch_size, -1, sin.size(1)).unsqueeze(1)  # [B, 1, S, D/2]
            cos = cos.view(batch_size, -1, cos.size(1)).unsqueeze(1)  # [B, 1, S, D/2]

            # 只对新计算的token应用RoPE
            if past_key_value is not None and self.config.kv_cache_enabled:
                # 为最后一个token创建sin/cos
                q_len = q.size(2)
                sin, cos = sin[:, :, -q_len:, :], cos[:, :, -q_len:, :]

            # 应用旋转位置嵌入
            q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        # --- 3. 注意力计算 ---
        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.attn_p if self.training else 0.0,
            is_causal=use_causal,
            scale=self.scale,
        )  # [B, N, S, D]

        # --- 4. 合并头部输出 ---
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)

        # --- 5. 输出投影和dropout ---
        output = self.dropout(self.out_proj(attn_output))

        # --- 6. 残差连接 ---
        output = output + residual

        # --- 7. Layer Normalization (Post-LN模式) ---
        if not self.norm_first:
            output = self.norm(output)

        # 返回输出和更新的KV缓存
        if use_cache and self.config.kv_cache_enabled:
            return output, (k, v)
        else:
            return output

def test_mha():
    # 基本用法
    config = MultiHeadAttentionConfig(
        hidden_size=512,
        num_heads=8,
        dropout=0.1,
        norm_first=True
    )

    mha = EnhancedMultiHeadAttention(config)

    # 单次前向传播
    x = torch.randn(2, 10, 512)  # [batch_size, seq_len, hidden_size]
    output = mha(x)

    # 使用KV缓存的自回归生成示例
    config = MultiHeadAttentionConfig(
        hidden_size=512,
        num_heads=8,
        dropout=0.1,
        kv_cache_enabled=True,
        norm_first=True
    )

    mha = EnhancedMultiHeadAttention(config)

    # 第一步：处理提示文本
    prompt = torch.randn(1, 5, 512)  # [batch_size, prompt_len, hidden_size]
    output, past_kv = mha(prompt, use_cache=True)

    # 自回归生成
    for i in range(10):
        # 只输入最后一个token
        next_token = output[:, -1:, :]  # [batch_size, 1, hidden_size]
        # 使用KV缓存
        output, past_kv = mha(
            next_token,
            past_key_value=past_kv,
            use_cache=True
        )
        # output现在包含新生成的token表示

def test_rope():
    # 配置使用RoPE的多头注意力
    config = MultiHeadAttentionConfig(
        hidden_size=512,
        num_heads=8,
        use_rotary_embeddings=True,
        rotary_dim=64  # 应用到每个头的前64维
    )

    mha = EnhancedMultiHeadAttention(config)

    # 输入和位置ID
    x = torch.randn(2, 10, 512)
    position_ids = torch.arange(10).unsqueeze(0).expand(2, 10)

    # 前向传播
    output = mha(x, position_ids=position_ids)

def test_separate_qkv():
    # 配置使用分离QKV投影的多头注意力
    config = MultiHeadAttentionConfig(
        hidden_size=768,
        num_heads=12,
        separate_qkv=True  # 使用分离的QKV投影
    )

    mha = EnhancedMultiHeadAttention(config)

    # 前向传播
    x = torch.randn(1, 20, 768)
    output = mha(x)
