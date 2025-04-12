import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class BaseAttention(nn.Module):
    """
    注意力机制的基类，定义共享接口和功能。
    所有特定注意力实现都应该继承此类。

    Args:
        hidden_size: 隐藏层维度大小
        dropout_p: Dropout概率 (default: 0.1)
        bias: 是否在线性层中使用偏置 (default: True)
        device: 模型所在设备
        dtype: 模型参数的数据类型
    """

    def __init__(
        self,
        hidden_size: int,
        dropout_p: float = 0.1,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.bias = bias
        self.device = device
        self.dtype = dtype

    def _init_weights(self, module: nn.Module, gain: float = 1.0):
        """权重初始化的共享方法"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """基类的前向传播方法"""
        raise NotImplementedError("子类必须实现forward方法")


class Attention(BaseAttention):
    """
    基础注意力机制实现 (缩放点积注意力)。

    Args:
        hidden_size: 隐藏层维度大小
        num_attention_heads: 注意力头数量 (default: 1)
        dropout_p: Dropout概率 (default: 0.1)
        bias: 是否使用偏置 (default: True)
        cross_attention: 是否支持跨注意力 (default: False)
        flash_attention: 是否启用Flash Attention (default: False)
        device: 模型所在设备
        dtype: 模型参数的数据类型
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 1,
        dropout_p: float = 0.1,
        bias: bool = True,
        cross_attention: bool = False,
        flash_attention: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(hidden_size, dropout_p, bias, device, dtype)

        self.num_attention_heads = num_attention_heads
        self.head_size = hidden_size // num_attention_heads
        self.scaling = 1.0 / math.sqrt(self.head_size)
        self.cross_attention = cross_attention

        # 检查Flash Attention可用性
        self.flash_attention = (
            flash_attention and hasattr(F, "scaled_dot_product_attention") and not cross_attention
        )  # 跨注意力通常不支持flash

        # 线性投影层
        self.query = nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)
        if cross_attention:
            self.key = nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)
            self.value = nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)
        else:
            # 自注意力可以合并投影
            self.in_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=bias, device=device, dtype=dtype)

        # 输出投影
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout_p)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        # 输出层使用较小的初始化
        self._init_weights(self.out_proj, gain=1.0 / math.sqrt(2.0))
        if self.cross_attention:
            self._init_weights(self.query)
            self._init_weights(self.key)
            self._init_weights(self.value)
        else:
            self._init_weights(self.in_proj)

    def _split_heads(self, x: Tensor, num_heads: int) -> Tensor:
        """分割多头"""
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, num_heads, self.head_size).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """合并多头"""
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        attention_mask: Tensor | None = None,
        use_cache: bool = False,
        past_key_value: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        前向传播

        Args:
            hidden_states: 输入张量 [batch_size, seq_len, hidden_size]
            encoder_hidden_states: 编码器输出 (跨注意力时使用)
            attention_mask: 注意力掩码 (True/1表示需要mask的位置)
            use_cache: 是否缓存KV
            past_key_value: 缓存的KV对

        Returns:
            输出张量或(输出张量, KV缓存)
        """
        # 确定是自注意力还是跨注意力
        is_cross_attention = encoder_hidden_states is not None and self.cross_attention

        batch_size, seq_len, _ = hidden_states.size()

        # 投影查询
        query_states = self.query(hidden_states)

        # 处理KV
        if is_cross_attention:
            key_states = self.key(encoder_hidden_states)
            value_states = self.value(encoder_hidden_states)
            kv_seq_len = encoder_hidden_states.size(1)
        else:
            # 自注意力 - 合并投影更高效
            q, k, v = self.in_proj(hidden_states).chunk(3, dim=-1)
            query_states, key_states, value_states = q, k, v
            kv_seq_len = seq_len

        # 处理增量解码
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)
            kv_seq_len = key_states.size(1)

        print(f"kv_seq_len: {kv_seq_len}")  # just to skip F841

        present_key_value = (key_states, value_states) if use_cache else None

        # 分割多头
        query_states = self._split_heads(query_states, self.num_attention_heads)
        key_states = self._split_heads(key_states, self.num_attention_heads)
        value_states = self._split_heads(value_states, self.num_attention_heads)

        # Flash Attention路径
        if self.flash_attention and not is_cross_attention and attention_mask is None:
            context = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,
            )
        else:
            # 传统注意力计算
            attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scaling

            # 应用掩码
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_probs = F.softmax(attn_weights, dim=-1)
            attn_probs = self.dropout(attn_probs)
            context = torch.matmul(attn_probs, value_states)

        # 合并多头并输出
        context = self._merge_heads(context)
        output = self.out_proj(context)

        return (output, present_key_value) if use_cache else output


class MultiHeadAttention(BaseAttention):
    """
    优化的多头注意力实现。

    Args:
        hidden_size: 隐藏层维度大小
        num_attention_heads: 注意力头数量 (default: 8)
        dropout_p: Dropout概率 (default: 0.1)
        bias: 是否使用偏置 (default: True)
        layer_norm_eps: LayerNorm epsilon (default: 1e-5)
        pre_norm: 是否使用Pre-LayerNorm (default: True)
        cross_attention: 是否支持跨注意力 (default: False)
        flash_attention: 是否启用Flash Attention (default: True)
        device: 模型所在设备
        dtype: 模型参数的数据类型
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 8,
        dropout_p: float = 0.1,
        bias: bool = True,
        layer_norm_eps: float = 1e-5,
        pre_norm: bool = True,
        cross_attention: bool = False,
        flash_attention: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(hidden_size, dropout_p, bias, device, dtype)

        assert hidden_size % num_attention_heads == 0, "hidden_size必须能被num_attention_heads整除"

        self.num_attention_heads = num_attention_heads
        self.head_size = hidden_size // num_attention_heads
        self.scaling = 1.0 / math.sqrt(self.head_size)
        self.pre_norm = pre_norm
        self.cross_attention = cross_attention

        # 检查Flash Attention可用性
        self.flash_attention = flash_attention and hasattr(F, "scaled_dot_product_attention") and not cross_attention

        # LayerNorm
        if pre_norm:
            self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps, device=device, dtype=dtype)
            if cross_attention:
                self.encoder_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps, device=device, dtype=dtype)

        # 投影层
        if cross_attention:
            self.query = nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)
            self.key = nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)
            self.value = nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)
        else:
            # 自注意力使用合并投影更高效
            self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=bias, device=device, dtype=dtype)

        # 输出层
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout_p)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        # 输出层使用较小的初始化
        self._init_weights(self.out_proj, gain=1.0 / math.sqrt(2.0))
        if self.cross_attention:
            self._init_weights(self.query)
            self._init_weights(self.key)
            self._init_weights(self.value)
        else:
            self._init_weights(self.qkv_proj)

    def _shape(self, x: Tensor, seq_len: int) -> Tensor:
        """重塑张量为多头形状"""
        return x.view(x.size(0), seq_len, self.num_attention_heads, self.head_size).transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        attention_mask: Tensor | None = None,
        use_cache: bool = False,
        past_key_value: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        前向传播

        Args:
            hidden_states: 输入张量 [batch_size, seq_len, hidden_size]
            encoder_hidden_states: 编码器输出 (跨注意力时使用)
            attention_mask: 注意力掩码 (True/1表示需要mask的位置)
            use_cache: 是否缓存KV
            past_key_value: 缓存的KV对

        Returns:
            输出张量或(输出张量, KV缓存)
        """
        residual = hidden_states
        is_cross_attention = encoder_hidden_states is not None and self.cross_attention

        # Pre-LayerNorm
        if self.pre_norm:
            hidden_states = self.layer_norm(hidden_states)
            if is_cross_attention:
                encoder_hidden_states = self.encoder_layer_norm(encoder_hidden_states)

        batch_size, seq_len, _ = hidden_states.size()

        # 处理查询
        query_states = (
            self.query(hidden_states) if is_cross_attention else self.qkv_proj(hidden_states)[:, :, : self.hidden_size]
        )

        # 处理KV
        if is_cross_attention:
            key_states = self.key(encoder_hidden_states)
            value_states = self.value(encoder_hidden_states)
            kv_seq_len = encoder_hidden_states.size(1)
        else:
            # 自注意力
            key_states = self.qkv_proj(hidden_states)[:, :, self.hidden_size : 2 * self.hidden_size]
            value_states = self.qkv_proj(hidden_states)[:, :, 2 * self.hidden_size :]
            kv_seq_len = seq_len

        # 处理增量解码
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)
            kv_seq_len = key_states.size(1)

        present_key_value = (key_states, value_states) if use_cache else None

        # 重塑为多头
        query_states = self._shape(query_states, seq_len)
        key_states = self._shape(key_states, kv_seq_len)
        value_states = self._shape(value_states, kv_seq_len)

        # Flash Attention路径
        if self.flash_attention and not is_cross_attention:
            if attention_mask is not None:
                # 转换mask格式为Flash Attention所需
                attention_mask = attention_mask.to(dtype=torch.bool)
            context = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,
            )
        else:
            # 传统注意力计算
            attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scaling

            # 应用掩码
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_probs = F.softmax(attn_weights, dim=-1)
            attn_probs = self.dropout(attn_probs)
            context = torch.matmul(attn_probs, value_states)

        # 合并多头并输出
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(context)
        output = self.dropout(output)

        # 残差连接
        if self.pre_norm:
            output = residual + output
        else:
            output = self.layer_norm(residual + output)

        return (output, present_key_value) if use_cache else output


class MultiLatentAttention(BaseAttention):
    """
    高效的多潜在注意力实现。

    Args:
        hidden_size: 输入隐藏层维度大小
        num_attention_heads: 注意力头数量 (default: 8)
        num_latents: 潜在向量数量 (default: 16)
        latent_size: 潜在向量大小 (default: None, 使用hidden_size)
        dropout_p: Dropout概率 (default: 0.1)
        bias: 是否使用偏置 (default: True)
        layer_norm_eps: LayerNorm epsilon (default: 1e-5)
        pre_norm: 是否使用Pre-LayerNorm (default: True)
        device: 模型所在设备
        dtype: 模型参数的数据类型
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 8,
        num_latents: int = 16,
        latent_size: int | None = None,
        dropout_p: float = 0.1,
        bias: bool = True,
        layer_norm_eps: float = 1e-5,
        pre_norm: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(hidden_size, dropout_p, bias, device, dtype)

        self.latent_size = latent_size if latent_size is not None else hidden_size
        self.num_latents = num_latents
        self.pre_norm = pre_norm

        assert hidden_size % num_attention_heads == 0, "hidden_size必须能被num_attention_heads整除"
        assert self.latent_size % num_attention_heads == 0, "latent_size必须能被num_attention_heads整除"

        self.num_attention_heads = num_attention_heads
        self.head_size = hidden_size // num_attention_heads
        self.latent_head_size = self.latent_size // num_attention_heads

        # LayerNorm
        if pre_norm:
            self.input_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps, device=device, dtype=dtype)
            self.latent_norm = nn.LayerNorm(self.latent_size, eps=layer_norm_eps, device=device, dtype=dtype)
            self.intermediate_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps, device=device, dtype=dtype)

        # 可学习的潜在向量
        self.latents = nn.Parameter(torch.randn(1, num_latents, self.latent_size, device=device, dtype=dtype))

        # 投影层
        # 阶段1: 潜在向量关注输入
        self.latent_to_q = nn.Linear(self.latent_size, hidden_size, bias=bias, device=device, dtype=dtype)
        self.input_to_kv = nn.Linear(hidden_size, 2 * hidden_size, bias=bias, device=device, dtype=dtype)

        # 阶段2: 输入关注潜在向量
        self.input_to_q = nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)
        self.latent_to_kv = nn.Linear(hidden_size, 2 * hidden_size, bias=bias, device=device, dtype=dtype)

        # 输出层
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout_p)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        # 潜在向量初始化
        nn.init.normal_(self.latents, std=0.02)

        # 投影层初始化
        self._init_weights(self.latent_to_q)
        self._init_weights(self.input_to_kv)
        self._init_weights(self.input_to_q)
        self._init_weights(self.latent_to_kv)

        # 输出层使用较小的初始化
        self._init_weights(self.out_proj, gain=1.0 / math.sqrt(2.0))

    def _shape(self, x: Tensor, is_latent: bool = False) -> Tensor:
        """重塑张量为多头形状"""
        head_size = self.latent_head_size if is_latent else self.head_size
        return x.view(x.size(0), -1, self.num_attention_heads, head_size).transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """
        前向传播

        Args:
            hidden_states: 输入张量 [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码 (True/1表示需要mask的位置)

        Returns:
            输出张量 [batch_size, seq_len, hidden_size]
        """
        residual = hidden_states
        batch_size = hidden_states.size(0)

        # 扩展潜在向量以匹配batch_size
        latents = self.latents.expand(batch_size, -1, -1)

        # Pre-LayerNorm
        if self.pre_norm:
            hidden_states = self.input_norm(hidden_states)
            latents = self.latent_norm(latents)

        # --- 阶段1: 潜在向量关注输入 ---
        # 投影
        q1 = self.latent_to_q(latents)
        k1, v1 = self.input_to_kv(hidden_states).chunk(2, dim=-1)

        # 多头处理
        q1 = self._shape(q1, is_latent=True)
        k1 = self._shape(k1)
        v1 = self._shape(v1)

        # 注意力计算
        attn_weights1 = torch.matmul(q1, k1.transpose(-1, -2)) / math.sqrt(self.head_size)

        # 应用掩码
        if attention_mask is not None:
            attn_weights1 = attn_weights1 + attention_mask

        attn_probs1 = F.softmax(attn_weights1, dim=-1)
        attn_probs1 = self.dropout(attn_probs1)

        # 计算上下文
        context1 = torch.matmul(attn_probs1, v1)
        context1 = context1.transpose(1, 2).contiguous().view(batch_size, -1, self.latent_size)

        # 更新潜在向量 (可选残差)
        intermediate_latents = context1

        # --- 阶段2: 输入关注潜在向量 ---
        # Pre-LayerNorm
        if self.pre_norm:
            intermediate_latents = self.intermediate_norm(intermediate_latents)

        # 投影
        q2 = self.input_to_q(hidden_states)
        k2, v2 = self.latent_to_kv(intermediate_latents).chunk(2, dim=-1)

        # 多头处理
        q2 = self._shape(q2)
        k2 = self._shape(k2, is_latent=True)
        v2 = self._shape(v2, is_latent=True)

        # 注意力计算
        attn_weights2 = torch.matmul(q2, k2.transpose(-1, -2)) / math.sqrt(self.head_size)
        attn_probs2 = F.softmax(attn_weights2, dim=-1)
        attn_probs2 = self.dropout(attn_probs2)

        # 计算上下文
        context2 = torch.matmul(attn_probs2, v2)
        context2 = context2.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)

        # 输出投影
        output = self.out_proj(context2)
        output = self.dropout(output)

        # 残差连接
        if self.pre_norm:
            output = residual + output
        else:
            output = self.input_norm(residual + output)

        return output
