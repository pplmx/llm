from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class AttentionConfig:
    """
    注意力机制的配置类。

    Args:
        # 核心模型参数
        hidden_size: 隐藏层维度，必须是num_heads的整数倍
        num_heads: 注意力头的数量，默认为8

        # 正则化相关参数
        dropout: 用于输出的dropout率，默认为0.1
        attention_dropout: 用于注意力权重的dropout率，默认与dropout相同
        eps: LayerNorm的epsilon值，用于数值稳定性，默认为1e-5

        # 功能开关参数（统一使用use_前缀）
        use_bias: 是否在线性投影中使用偏置，默认为False
        use_norm_first: 是否先进行层归一化（Pre-LN），默认为True
        use_causal_mask: 是否使用因果注意力掩码，默认为False
        use_separate_qkv: 是否使用分离的QKV投影，默认为False
        use_rotary_embeddings: 是否使用旋转位置嵌入(RoPE)，默认为False
        use_kv_cache: 是否启用KV缓存用于生成，默认为False

        # 特殊参数
        rotary_dim: 旋转位置嵌入的维度，默认为head_dim

        # 技术细节参数
        device: 张量设备
        dtype: 张量数据类型
    """

    # 核心模型参数
    hidden_size: int
    num_heads: int = 8

    # 正则化相关参数
    dropout: float = 0.1
    attention_dropout: float | None = None
    eps: float = 1e-5

    # 架构参数
    use_norm_first: bool = True
    use_bias: bool = False
    use_causal_mask: bool = False
    use_separate_qkv: bool = False

    # 位置编码相关参数
    use_rotary_embeddings: bool = False
    rotary_dim: int | None = None

    # 推理相关参数
    use_kv_cache: bool = False

    # 硬件相关参数
    device: torch.device | str | None = None
    dtype: torch.dtype | None = None

    def __post_init__(self):
        """初始化后的验证和设置默认值"""
        # 验证配置
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})")

        # 设置默认值
        if self.attention_dropout is None:
            self.attention_dropout = self.dropout

        if self.use_rotary_embeddings and self.rotary_dim is None:
            # 默认使用每个头的全部维度
            self.rotary_dim = self.hidden_size // self.num_heads


class EnhancedMultiHeadAttention(nn.Module):
    """
    增强的多头注意力实现，使用配置类进行初始化。

    这个实现支持多种高级功能:
    - 旋转位置嵌入 (RoPE)：为模型提供相对位置信息而不需要位置编码
    - KV缓存：加速自回归生成过程
    - 分离或融合的QKV投影：可以选择更灵活的架构
    - 可配置的归一化策略：Pre-LN或Post-LN

    多头注意力机制工作原理：
    1. 将输入投影到查询(Q)、键(K)和值(V)
    2. 将Q、K、V分割为多个头
    3. 每个头独立计算注意力分数并获取加权值
    4. 合并所有头的输出
    5. 进行最终的投影和残差连接
    """

    def __init__(self, config: AttentionConfig):
        """
        初始化多头注意力模块。

        Args:
            config: 包含所有配置参数的MultiHeadAttentionConfig对象
        """
        super().__init__()
        self.config = config

        # 设置基本参数
        factory_kwargs = {"device": config.device, "dtype": config.dtype}
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        # 配置选项
        self.norm_first = config.use_norm_first
        self.is_causal = config.use_causal_mask
        self.p = config.dropout
        self.attn_p = config.attention_dropout
        self.use_rope = config.use_rotary_embeddings

        if self.use_rope:
            self.rotary_dim = config.rotary_dim

        # 层归一化
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.eps, **factory_kwargs)

        # QKV投影策略
        if config.use_separate_qkv:
            # 分离的QKV投影（每个都有自己的权重矩阵）
            self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias, **factory_kwargs)
            self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias, **factory_kwargs)
            self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias, **factory_kwargs)
            self.has_separate_qkv = True
        else:
            # 融合的QKV投影（单个权重矩阵）
            self.qkv_proj = nn.Linear(
                config.hidden_size, 3 * config.hidden_size, bias=config.use_bias, **factory_kwargs
            )
            self.has_separate_qkv = False

        # 输出投影和dropout
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias, **factory_kwargs)
        self.dropout = nn.Dropout(config.dropout)

    def _rotate_half(self, x: Tensor) -> Tensor:
        """
        旋转向量的一半维度。

        RoPE的基础操作，对输入张量的前半部分和后半部分进行特殊的旋转变换。

        Args:
            x: 输入张量，最后一个维度是要旋转的维度

        Returns:
            旋转后的张量
        """
        # x shape: [..., D_r]
        x1 = x[..., : self.rotary_dim // 2]  # 前半部分
        x2 = x[..., self.rotary_dim // 2 :]  # 后半部分
        # 旋转操作：交换前后半部分并在第二部分加负号
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
        """
        应用旋转位置嵌入 (RoPE)。

        RoPE通过旋转操作在Q和K中注入相对位置信息，无需额外的位置编码。
        只应用于每个头的前rotary_dim维度。

        Args:
            q: 查询张量 [B, N, S, D]
            k: 键张量 [B, N, S, D]
            cos: 余弦位置编码 [B, 1, S, D_r]
            sin: 正弦位置编码 [B, 1, S, D_r]

        Returns:
            应用RoPE后的查询和键张量
        """
        # 分离需要旋转和不需要旋转的部分
        q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
        k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]

        # 应用旋转编码
        # cos/sin形状[B, 1, S, D_r]可广播到q_rot/k_rot的[B, N, S, D_r]
        q_rotated = (q_rot * cos) + (self._rotate_half(q_rot) * sin)
        k_rotated = (k_rot * cos) + (self._rotate_half(k_rot) * sin)

        # 合并旋转和非旋转部分
        q_out = torch.cat((q_rotated, q_pass), dim=-1)
        k_out = torch.cat((k_rotated, k_pass), dim=-1)

        return q_out, k_out

    def _reshape_for_attention(self, x: Tensor, batch_size: int, seq_len: int) -> Tensor:
        """
        将投影后的张量重新整形为注意力计算所需的形状。

        Args:
            x: 输入张量 [B, S, H] 或 [B, S, N*D]
            batch_size: 批量大小
            seq_len: 序列长度

        Returns:
            重新整形的张量 [B, N, S, D]，适合注意力计算
        """
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _get_qkv(
        self, hidden_states: Tensor, batch_size: int, seq_len: int, past_key_value: tuple[Tensor, Tensor] | None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        根据配置计算查询(Q)、键(K)和值(V)张量。

        处理分离/融合QKV投影和KV缓存的逻辑。

        Args:
            hidden_states: 输入隐藏状态 [B, S, H]
            batch_size: 批量大小
            seq_len: 序列长度
            past_key_value: 可选的之前的KV缓存

        Returns:
            查询、键和值张量，形状均为 [B, N, S, D]
        """
        use_cache = past_key_value is not None and self.config.use_kv_cache

        if self.has_separate_qkv:
            # 使用分离的QKV投影
            q = self.q_proj(hidden_states)  # [B, S, H]

            if use_cache:
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
            if use_cache:
                # 对所有token计算Q，但只对新token计算KV
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
                # 常规QKV投影计算
                qkv = self.qkv_proj(hidden_states)  # [B, S, 3*H]
                qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
                q, k, v = [qkv[:, :, i].transpose(1, 2) for i in range(3)]  # 每个 [B, N, S, D]

        # 确保所有张量都是正确的形状 [B, N, S, D]
        if not self.has_separate_qkv or not use_cache:
            q = self._reshape_for_attention(q, batch_size, seq_len) if q.dim() == 3 else q
            k = self._reshape_for_attention(k, batch_size, k.size(1)) if k.dim() == 3 else k
            v = self._reshape_for_attention(v, batch_size, v.size(1)) if v.dim() == 3 else v

        return q, k, v

    def _compute_rope_sincos(self, position_ids: Tensor, batch_size: int, seq_length: int) -> tuple[Tensor, Tensor]:
        """
        计算RoPE所需的正弦和余弦值。

        Args:
            position_ids: 位置ID张量 [B, S]
            batch_size: 批量大小
            seq_length: 序列长度

        Returns:
            sin, cos: 正弦和余弦张量，形状为 [B, 1, S, D_r]
        """
        position_ids = position_ids.view(-1, seq_length)

        # 生成角度，考虑到旋转位置嵌入的维度
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.rotary_dim, 2, device=position_ids.device) / self.rotary_dim))
        sincos = torch.einsum("i,j->ij", position_ids.flatten(), inv_freq)
        sin, cos = sincos.sin(), sincos.cos()

        # 重塑成正确的形状
        # 重塑 sin 和 cos: [B, S, D_r/2]
        sin = sin.view(batch_size, seq_length, -1)
        cos = cos.view(batch_size, seq_length, -1)

        # 扩展最后一个维度以匹配 rotary_dim: [B, S, D_r]
        sin = sin.repeat_interleave(2, dim=-1)
        cos = cos.repeat_interleave(2, dim=-1)

        # 添加 head 维度（用于广播）: [B, 1, S, D_r]
        sin = sin.unsqueeze(1)
        cos = cos.unsqueeze(1)

        return sin, cos

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
        多头注意力的前向传播。

        处理流程:
        1. 应用层归一化（如果norm_first=True）
        2. 计算查询(Q)、键(K)和值(V)
        3. 应用旋转位置嵌入（如果启用）
        4. 计算注意力权重并获取加权值
        5. 合并多头输出并进行输出投影
        6. 应用dropout和残差连接
        7. 应用层归一化（如果norm_first=False）

        Args:
            hidden_states: 输入隐藏状态，形状为 [B, S, H]
            attn_mask: 可选的注意力掩码
            is_causal: 是否使用因果掩码，覆盖默认设置
            position_ids: 可选的位置ID，用于旋转位置嵌入
            past_key_value: 可选的之前计算的KV缓存
            use_cache: 是否返回更新的KV缓存

        Returns:
            如果use_cache=False: 输出张量，形状为 [B, S, H]
            如果use_cache=True: (输出张量, (键缓存, 值缓存))
        """
        # 获取基本维度信息
        batch_size, seq_len, _ = hidden_states.size()

        # 保存残差连接用于最后添加
        residual = hidden_states

        # 确定此次调用的因果设置
        use_causal = self.is_causal if is_causal is None else is_causal

        # --- 1. 层归一化 (Pre-LN模式) ---
        if self.norm_first:
            hidden_states = self.norm(hidden_states)

        # --- 2. 计算Q、K、V并整形 ---
        q, k, v = self._get_qkv(hidden_states, batch_size, seq_len, past_key_value)

        # --- 3. 应用旋转位置嵌入（如果启用）---
        if self.use_rope and position_ids is not None:
            # 计算正弦和余弦值
            sin, cos = self._compute_rope_sincos(position_ids, batch_size, seq_len)

            # 如果使用KV缓存，只对新token应用RoPE
            if past_key_value is not None and self.config.use_kv_cache:
                q_len = q.size(2)
                sin, cos = sin[:, :, -q_len:, :], cos[:, :, -q_len:, :]

            # 应用旋转位置嵌入
            q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        # --- 4. 注意力计算 ---
        # 使用PyTorch的原生scaled_dot_product_attention优化实现
        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.attn_p if self.training else 0.0,
            is_causal=use_causal,
        )  # [B, N, S, D]

        # --- 5. 合并头部输出 ---
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)

        # --- 6. 输出投影和dropout ---
        output = self.dropout(self.out_proj(attn_output))

        # --- 7. 残差连接 ---
        output = output + residual

        # --- 8. 层归一化 (Post-LN模式) ---
        if not self.norm_first:
            output = self.norm(output)

        # 返回输出和更新的KV缓存
        if use_cache and self.config.use_kv_cache:
            return output, (k, v)
        else:
            return output


# 示例用法
def test_mha():
    """基本用法示例"""
    # 创建配置
    config = AttentionConfig(
        hidden_size=512,  # 隐藏层维度
        num_heads=8,  # 注意力头数量
        dropout=0.1,  # dropout比率
        use_norm_first=True,  # 使用Pre-LN结构
    )

    # 初始化多头注意力模块
    mha = EnhancedMultiHeadAttention(config)

    # 单次前向传播
    x = torch.randn(2, 10, 512)  # [batch_size, seq_len, hidden_size]
    output = mha(x)
    print(f"输入形状: {x.shape}, 输出形状: {output.shape}")


def test_kv_cache():
    """使用KV缓存的自回归生成示例"""
    # 创建启用KV缓存的配置
    config = AttentionConfig(
        hidden_size=512,
        num_heads=8,
        dropout=0.1,
        use_kv_cache=True,  # 启用KV缓存
        use_norm_first=True,
    )

    mha = EnhancedMultiHeadAttention(config)

    # 第一步：处理提示文本
    prompt = torch.randn(1, 5, 512)  # [batch_size, prompt_len, hidden_size]
    print(f"提示文本形状: {prompt.shape}")

    # 获取初始输出和KV缓存
    output, past_kv = mha(prompt, use_cache=True)
    print(f"初始输出形状: {output.shape}")
    print(f"KV缓存形状: {past_kv[0].shape}, {past_kv[1].shape}")


def test_rotary_embeddings():
    """测试旋转位置嵌入（RoPE）功能"""
    config = AttentionConfig(
        hidden_size=512,
        num_heads=8,
        dropout=0.1,
        use_rotary_embeddings=True,  # 启用旋转位置嵌入
        rotary_dim=64,  # 指定旋转位置嵌入的维度
        use_norm_first=True,
    )

    mha = EnhancedMultiHeadAttention(config)

    # 创建输入和位置ID
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, 512)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    # 使用RoPE的前向传播
    output = mha(hidden_states=x, position_ids=position_ids)

    print(f"RoPE测试 - 输入形状: {x.shape}, 输出形状: {output.shape}")
    print(f"RoPE维度: {config.rotary_dim}, 位置ID形状: {position_ids.shape}")

    # 验证输出形状和输入相同
    assert output.shape == x.shape, "RoPE输出形状应与输入相同"


def test_causal_masking():
    """测试因果掩码（Causal Masking）功能"""
    config = AttentionConfig(
        hidden_size=512,
        num_heads=8,
        dropout=0.1,
        use_causal_mask=True,  # 启用因果掩码
        use_norm_first=True,
    )

    mha = EnhancedMultiHeadAttention(config)

    # 创建输入
    batch_size, seq_len = 2, 16
    x = torch.randn(batch_size, seq_len, 512)

    # 前向传播
    output = mha(x)

    print(f"因果掩码测试 - 输入形状: {x.shape}, 输出形状: {output.shape}")

    # 验证输出形状和输入相同
    assert output.shape == x.shape, "因果掩码输出形状应与输入相同"

    # 测试覆盖因果掩码设置
    mha(x, is_causal=False)
    print("成功用参数覆盖了默认的因果掩码设置")


def test_separate_qkv():
    """测试分离的QKV投影"""
    config = AttentionConfig(
        hidden_size=512,
        num_heads=8,
        dropout=0.1,
        use_separate_qkv=True,  # 使用分离的QKV投影
        use_norm_first=False,  # 测试Post-LN结构
    )

    mha = EnhancedMultiHeadAttention(config)

    # 创建输入
    batch_size, seq_len = 3, 12
    x = torch.randn(batch_size, seq_len, 512)

    # 前向传播
    output = mha(x)

    print(f"分离QKV测试 - 输入形状: {x.shape}, 输出形状: {output.shape}")
    print(f"使用Post-LN结构: {not config.use_norm_first}")

    # 验证输出形状和输入相同
    assert output.shape == x.shape, "分离QKV输出形状应与输入相同"


def test_rope_with_kv_cache():
    """测试同时使用旋转位置嵌入和KV缓存"""
    config = AttentionConfig(
        hidden_size=512,
        num_heads=8,
        dropout=0.1,
        use_rotary_embeddings=True,  # 启用旋转位置嵌入
        use_kv_cache=True,  # 启用KV缓存
        use_norm_first=True,
    )

    mha = EnhancedMultiHeadAttention(config)

    # 第一步：处理提示文本
    batch_size, prompt_len = 1, 8
    prompt = torch.randn(batch_size, prompt_len, 512)
    position_ids = torch.arange(prompt_len).unsqueeze(0)

    # 获取初始输出和KV缓存
    output, past_kv = mha(prompt, position_ids=position_ids, use_cache=True)

    print(f"RoPE+KV缓存测试 - 提示形状: {prompt.shape}, 初始输出形状: {output.shape}")
    print(f"初始KV缓存形状: {past_kv[0].shape}, {past_kv[1].shape}")

    # 模拟自回归生成
    for i in range(5):
        # 新token输入
        new_token = torch.randn(batch_size, 1, 512)
        # 更新位置ID
        new_position_id = torch.tensor([[prompt_len + i]])

        # 前向传播，使用缓存
        output, past_kv = mha(new_token, position_ids=new_position_id, past_key_value=past_kv, use_cache=True)

        print(f"步骤 {i + 1} - 输出形状: {output.shape}")
        print(f"更新后KV缓存形状: {past_kv[0].shape}, {past_kv[1].shape}")


def test_attention_mask():
    """测试自定义注意力掩码"""
    config = AttentionConfig(
        hidden_size=512,
        num_heads=8,
        dropout=0.1,
        use_causal_mask=False,  # 不使用默认因果掩码
        use_norm_first=True,
    )

    mha = EnhancedMultiHeadAttention(config)

    # 创建输入
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, 512)

    # 创建自定义注意力掩码（例如：填充掩码）
    # 假设第二个序列的最后两个token是填充的
    attn_mask = torch.ones(batch_size, 1, seq_len, seq_len)
    attn_mask[1, :, :, 8:] = 0  # 第二个序列的最后两个位置被掩盖
    attn_mask = attn_mask == 0  # 转换为布尔掩码，True表示被掩盖的位置

    # 前向传播，使用自定义掩码
    output = mha(x, attn_mask=attn_mask)

    print(f"注意力掩码测试 - 输入形状: {x.shape}, 输出形状: {output.shape}")
    print(f"掩码形状: {attn_mask.shape}, 掩码类型: {attn_mask.dtype}")


def test_long_sequence():
    """测试处理长序列的能力"""
    config = AttentionConfig(
        hidden_size=256,
        num_heads=4,
        dropout=0.1,
        use_norm_first=True,
    )

    mha = EnhancedMultiHeadAttention(config)

    # 创建较长序列的输入
    batch_size, seq_len = 1, 2048
    x = torch.randn(batch_size, seq_len, 256)

    # 前向传播
    output = mha(x)

    print(f"长序列测试 - 输入形状: {x.shape}, 输出形状: {output.shape}")


if __name__ == "__main__":
    pytest.main([__file__])
