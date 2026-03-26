# Flash Attention 集成设计

**Date**: 2026-03-26
**Author**: AI Assistant
**Status**: Draft

## 目标

为 LLM 项目集成 Flash Attention 2，通过分层后端架构提升注意力计算性能。

## 背景

当前使用 PyTorch 的 `F.scaled_dot_product_attention`，Flash Attention 可提供 2-3x 性能提升。

## 设计原则

1. **分层后端** - 后端独立，互不感知
2. **自动回退** - 不可用时自动使用 PyTorch SDPA
3. **保持 API** - 现有调用方无需修改
4. **延迟加载** - 避免强制依赖

## 架构

```
mha.py (MultiHeadAttention)
    │
    ├── backend.py (后端选择逻辑)
    │       │
    │       ├── torch_sdpa()  ← 现有 PyTorch SDPA
    │       │
    │       └── flash_attn()  ← 新增 Flash Attention
    │
    └── 输出统一格式: [B, N, S, D] → [B, S, H]
```

## 文件结构

```
src/llm/core/attn/
├── __init__.py
├── mha.py                 # MultiHeadAttention，调用 backend
├── sdpa.py                # 重命名为 torch_sdpa.py
├── flash_attn.py          # 新增: Flash Attention 后端
└── backend.py             # 新增: 后端选择逻辑
```

## 接口定义

### backend.py

```python
def get_attention_backend(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    window_size: int | None = None,
) -> str:
    """
    选择最佳后端: 'flash' 或 'torch'
    
    Returns:
        'flash': 使用 Flash Attention
        'torch': 使用 PyTorch SDPA
    """
```

### flash_attn.py

```python
def flash_attention(
    query: Tensor,   # [B, N, S, D]
    key: Tensor,     # [B, N, S, D]
    value: Tensor,   # [B, N, S, D]
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> Tensor:
    """
    Flash Attention 2 后端
    
    Args:
        query, key, value: [B, N, S, D]
        dropout_p: dropout 概率
        is_causal: 是否 causal
    
    Returns:
        output: [B, N, S, D] (统一格式，与 PyTorch SDPA 相同)
    """
```

### torch_sdpa.py (原 sdpa.py)

```python
def torch_sdpa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    window_size: int | None = None,
) -> Tensor:
    """PyTorch SDPA 后端，保持现有实现"""
```

## 实现细节

### 1. Flash Attention 输出处理

Flash Attention 输出 `[B, S, H]`，需要转换为 `[B, N, S, D]`：

```python
def flash_attention(query, key, value, ...):
    # 输入: [B, N, S, D]
    # Flash 需要: [B, S, N, D]
    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()
    
    # 调用 flash_attn
    out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=is_causal)
    
    # 输出: [B, S, N, D] → 转回 [B, N, S, D]
    return out.transpose(1, 2)
```

### 2. 后端选择逻辑

```python
def get_attention_backend(...):
    # 1. 检查 CUDA 可用
    if not query.is_cuda:
        return 'torch'
    
    # 2. 检查 Flash Attention 可用
    try:
        import flash_attn
    except ImportError:
        return 'torch'
    
    # 3. 检查是否支持当前场景
    if attn_mask is not None:
        return 'torch'  # Flash 不支持自定义 mask
    if window_size is not None:
        return 'torch'  # Flash 不支持 sliding window
    
    return 'flash'
```

### 3. MHA 集成

```python
# mha.py 修改
from llm.core.attn.backend import get_attention_backend
from llm.core.attn.torch_sdpa import torch_sdpa
from llm.core.attn.flash_attn import flash_attention

def forward(self, hidden_states, ...):
    ...
    # 1. 计算 QKV
    q, k, v = self._compute_qkv(x_for_qkv)
    
    # 2. 选择后端
    backend = get_attention_backend(q, k, v, attn_mask, self.p, use_causal, self.window_size)
    
    # 3. 执行注意力
    if backend == 'flash':
        attn_output = flash_attention(q, k, v, dropout_p, is_causal)
    else:
        attn_output = torch_sdpa(q, k, v, attn_mask, ..., window_size)
    
    # 4. 后续处理 (不变)
    attn_output = attn_output.transpose(1, 2).reshape(...)
```

## 依赖

```toml
# pyproject.toml
dependencies = [
    # Flash Attention (可选)
    # flash-attn>=2.0.0  # 注释说明为可选
]
```

## 测试

### 单元测试

```python
# tests/core/attn/test_flash_attention.py

def test_flash_attention_output_shape():
    """验证输出 shape 正确"""
    q = torch.randn(2, 4, 8, 32)  # [B, N, S, D]
    out = flash_attention(q, q, q)
    assert out.shape == (2, 4, 8, 32)

def test_flash_attention_output_value():
    """验证输出数值正确 (与 PyTorch SDPA 对比)"""
    torch.manual_seed(42)
    q = torch.randn(2, 4, 8, 32)
    
    out_flash = flash_attention(q, q, q, is_causal=True)
    out_torch = torch_sdpa(q, q, q, is_causal=True)
    
    # 允许一定误差
    assert torch.allclose(out_flash, out_torch, atol=1e-2)

def test_backend_selection():
    """验证后端选择正确"""
    # CUDA 可用时选择 flash
    # 有 mask 时选择 torch
    ...
```

## 后续扩展

1. **Variable Length SEQUENCE** - 使用 `flash_attn_varlen_func`
2. **Ring Attention** - 多 GPU 分布式
3. **配置选项** - 强制使用指定后端

## 风险

| 风险 | 缓解 |
|------|------|
| Flash Attention 安装困难 | 保持自动回退 |
| 版本兼容性 | 检测版本，版本不兼容时回退 |
| 数值精度差异 | 允许误差范围测试 |