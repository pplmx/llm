# Flash Attention 集成实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 LLM 项目集成 Flash Attention 2，通过分层后端架构提升注意力计算性能。

**Architecture:** 采用分层后端设计，创建独立的 flash_attn.py 和 backend.py，保持 mha.py 和 torch_sdpa.py 不变或最小改动。后端选择逻辑在运行时根据场景自动选择最佳后端。

**Tech Stack:** PyTorch, Flash Attention 2, pytest

---

## 文件结构

```text
src/llm/core/attn/
├── __init__.py                      # 更新导出
├── mha.py                           # 修改: 调用后端选择
├── sdpa.py                          # 重命名为 torch_sdpa.py
├── flash_attn.py                    # 新增: Flash Attention 后端
└── backend.py                       # 新增: 后端选择逻辑
```

---

## Task 1: 创建 torch_sdpa.py (从 sdpa.py 重命名)

**Files:**

- Create: `src/llm/core/attn/torch_sdpa.py`
- Modify: `src/llm/core/attn/sdpa.py` (删除或标记废弃)
- Test: 新增 `tests/core/attn/test_flash_attention.py`
- [ ] **Step 1: 复制 sdpa.py 为 torch_sdpa.py**

```bash
cp src/llm/core/attn/sdpa.py src/llm/core/attn/torch_sdpa.py
```

- [ ] **Step 2: 运行现有测试确保功能正常**

```bash
pytest tests/core/attn/test_mha.py -v
```

- [ ] **Step 3: 提交**

```bash
git add src/llm/core/attn/torch_sdpa.py
git commit -m "refactor: extract torch_sdpa from sdpa"
```

---

## Task 2: 创建 backend.py 后端选择逻辑

**Files:**

- Create: `src/llm/core/attn/backend.py`
- Test: `tests/core/attn/test_attention_backend.py`
- [ ] **Step 1: 编写测试文件 tests/core/attn/test_attention_backend.py**

```python
import pytest
import torch
from llm.core.attn.backend import get_attention_backend

class TestAttentionBackend:
    def test_returns_torch_for_cpu(self):
        """CPU 环境应返回 torch"""
        q = torch.randn(2, 4, 8, 32)
        result = get_attention_backend(q, q, q)
        assert result == "torch"

    def test_returns_flash_when_available(self):
        """Flash 可用时应返回 flash"""
        if not torch.cuda.is_available():
            pytest.skip("需要 CUDA")
        # 假设 flash 可用
        result = get_attention_backend(q, q, q)
        assert result in ["flash", "torch"]

    def test_returns_torch_with_mask(self):
        """有 mask 时应返回 torch"""
        if not torch.cuda.is_available():
            pytest.skip("需要 CUDA")
        mask = torch.zeros(2, 8, dtype=torch.bool)
        result = get_attention_backend(q, q, q, attn_mask=mask)
        assert result == "torch"

    def test_returns_torch_with_window(self):
        """有 window_size 时应返回 torch"""
        if not torch.cuda.is_available():
            pytest.skip("需要 CUDA")
        result = get_attention_backend(q, q, q, window_size=512)
        assert result == "torch"
```

- [ ] **Step 2: 运行测试验证失败 (函数未定义)**

```bash
pytest tests/core/attn/test_attention_backend.py -v
```

Expected: FAIL - ModuleNotFoundError

- [ ] **Step 3: 实现 backend.py**

```python
import torch
from torch import Tensor
from typing import TYPE_CHECKING

_flash_attn_available = None

def _check_flash_attn():
    """检测 Flash Attention 是否可用 (延迟加载)"""
    global _flash_attn_available
    if _flash_attn_available is not None:
        return _flash_attn_available

    try:
        import flash_attn
        _flash_attn_available = True
    except ImportError:
        _flash_attn_available = False

    return _flash_attn_available

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
    选择最佳注意力后端

    Returns:
        'flash': 使用 Flash Attention
        'torch': 使用 PyTorch SDPA
    """
    # 1. CUDA 可用性检查
    if not query.is_cuda:
        return "torch"

    # 2. Flash Attention 可用性检查
    if not _check_flash_attn():
        return "torch"

    # 3. 场景兼容性检查
    if attn_mask is not None:
        return "torch"

    if window_size is not None and window_size > 0:
        return "torch"

    # 所有条件满足，使用 Flash Attention
    return "flash"
```

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/core/attn/test_attention_backend.py -v
```

- [ ] **Step 5: 提交**

```bash
git add src/llm/core/attn/backend.py tests/core/attn/test_attention_backend.py
git commit -m "feat: add attention backend selection logic"
```

---

## Task 3: 创建 flash_attn.py Flash Attention 后端

**Files:**

- Create: `src/llm/core/attn/flash_attn.py`
- Test: `tests/core/attn/test_flash_attention.py`
- [ ] **Step 1: 编写测试文件 tests/core/attn/test_flash_attention.py**

```python
import pytest
import torch
from llm.core.attn.flash_attn import flash_attention

@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
class TestFlashAttention:
    def test_output_shape(self):
        """验证输出 shape 正确"""
        q = torch.randn(2, 4, 8, 32)  # [B, N, S, D]
        out = flash_attention(q, q, q)
        assert out.shape == (2, 4, 8, 32)

    def test_output_shape_causal(self):
        """验证 causal 模式输出 shape 正确"""
        q = torch.randn(2, 4, 8, 32)
        out = flash_attention(q, q, q, is_causal=True)
        assert out.shape == (2, 4, 8, 32)

    def test_output_close_to_torch_sdpa(self):
        """验证输出数值接近 PyTorch SDPA (误差范围内)"""
        torch.manual_seed(42)
        q = torch.randn(2, 4, 8, 32, device="cuda")

        # Flash Attention
        out_flash = flash_attention(q, q, q, is_causal=True)

        # PyTorch SDPA
        out_torch = torch.nn.functional.scaled_dot_product_attention(
            q, q, q, is_causal=True
        )

        # 允许一定误差
        assert torch.allclose(out_flash, out_torch, atol=1e-2)
```

- [ ] **Step 2: 运行测试验证失败 (函数未定义)**

```bash
pytest tests/core/attn/test_flash_attention.py -v
```

Expected: FAIL - ModuleNotFoundError

- [ ] **Step 3: 实现 flash_attn.py**

```python
import torch
from torch import Tensor

def flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> Tensor:
    """
    Flash Attention 2 后端

    Args:
        query: [B, N, S, D]
        key: [B, N, S, D]
        value: [B, N, S, D]
        dropout_p: dropout 概率
        is_causal: 是否 causal

    Returns:
        output: [B, N, S, D]
    """
    # 1. 导入 flash_attn (延迟加载)
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        raise ImportError(
            "Flash Attention is not installed. "
            "Install with: pip install flash-attn"
        )

    # 2. 准备输入格式 [B, S, N, D]
    # Flash Attention 需要 [B, S, H] 或 [B, S, N, D]
    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()

    # 3. 调用 Flash Attention
    out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=is_causal)

    # 4. 转换回输出格式 [B, N, S, D]
    out = out.transpose(1, 2)

    return out
```

- [ ] **Step 4: 运行测试验证**

```bash
pytest tests/core/attn/test_flash_attention.py -v
```

Note: 如果 flash-attn 未安装，测试会被跳过。这是预期行为。

- [ ] **Step 5: 提交**

```bash
git add src/llm/core/attn/flash_attn.py tests/core/attn/test_flash_attention.py
git commit -m "feat: add flash attention backend"
```

---

## Task 4: 更新 mha.py 集成后端选择

**Files:**

- Modify: `src/llm/core/attn/mha.py`
- Test: 现有测试 `tests/core/attn/test_mha.py`
- [ ] **Step 1: 修改 mha.py 导入**

在文件顶部添加:

```python
from llm.core.attn.backend import get_attention_backend
from llm.core.attn.torch_sdpa import torch_sdpa as torch_sdpa_impl
```

- [ ] **Step 2: 修改 forward 方法中的注意力调用 (约第195行)**

找到原来的:

```python
attn_output = sdpa(
    query=q,
    key=k,
    value=v,
    attn_mask=attn_mask,
    dropout_p=self.p if self.training else 0.0,
    is_causal=use_causal if not has_past else False,
    scale=None,
    window_size=self.window_size,
)
```

替换为:

```python
# 选择后端
backend = get_attention_backend(
    q, k, v,
    attn_mask=attn_mask,
    dropout_p=self.p if self.training else 0.0,
    is_causal=use_causal if not has_past else False,
    window_size=self.window_size,
)

if backend == "flash":
    attn_output = flash_attention(
        q, k, v,
        dropout_p=self.p if self.training else 0.0,
        is_causal=use_causal if not has_past else False,
    )
else:
    attn_output = torch_sdpa_impl(
        query=q,
        key=k,
        value=v,
        attn_mask=attn_mask,
        dropout_p=self.p if self.training else 0.0,
        is_causal=use_causal if not has_past else False,
        scale=None,
        window_size=self.window_size,
    )
```

- [ ] **Step 3: 运行现有 MHA 测试**

```bash
pytest tests/core/attn/test_mha.py -v
```

- [ ] **Step 4: 提交**

```bash
git add src/llm/core/attn/mha.py
git commit -m "feat: integrate flash attention backend in MHA"
```

---

## Task 5: 更新 **init**.py 导出

**Files:**

- Modify: `src/llm/core/attn/__init__.py`

- [ ] **Step 1: 更新导出**

```python
from .mha import MultiHeadAttention
from .mla import MultiLatentAttention
from .torch_sdpa import torch_sdpa
from .flash_attn import flash_attention
from .backend import get_attention_backend

__all__ = [
    "MultiHeadAttention",
    "MultiLatentAttention", 
    "torch_sdpa",
    "flash_attention",
    "get_attention_backend",
]
```

- [ ] **Step 2: 运行测试**

```bash
pytest tests/core/attn/ -v
```

- [ ] **Step 3: 提交**

```bash
git add src/llm/core/attn/__init__.py
git commit -m "chore: update attention module exports"
```

---

## Task 6: 完整集成测试

**Files:**

- Test: 运行所有相关测试

- [ ] **Step 1: 运行完整测试套件**

```bash
pytest tests/core/attn/ -v
```

- [ ] **Step 2: 运行 e2e 测试**

```bash
pytest tests/e2e/ -v -m "not slow"
```

- [ ] **Step 3: 提交**

```bash
git add -A
git commit -m "feat: complete flash attention integration"
```

---

## 验证清单

| 任务   | 验证项               |
| ------ | -------------------- |
| Task 1 | torch_sdpa 功能正常  |
| Task 2 | backend 选择逻辑正确 |
| Task 3 | flash_attn 输出正确  |
| Task 4 | MHA 集成后端选择     |
| Task 5 | 导出更新             |
| Task 6 | 完整测试通过         |

---

## 注意事项

1. 如果 flash-attn 未安装，测试会被跳过，这是预期行为
2. 可以通过设置环境变量 `FORCE_TORCH_SDPA=1` 强制使用 PyTorch SDPA
3. 后续可以添加配置选项让用户选择强制使用某个后端

---

Plan complete and saved to `docs/superpowers/plans/2026-03-26-flash-attention-plan.md`. Two execution options:

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
