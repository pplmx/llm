# 6. Use QLoRA for Memory-Efficient Fine-Tuning

Date: 2026-01-08

## Status

Accepted

## Context

大型语言模型的微调面临显著的显存限制：

1. **Full Fine-Tuning**: 7B 模型需要 ~28GB VRAM (fp16 weights + gradients)
2. **LoRA**: 减少训练参数，但基础权重仍占 ~14GB
3. **限制**: 消费级 GPU (8-16GB) 无法微调中大型模型

## Decision

实现 QLoRA（`src/llm/core/qlora.py`），结合量化和 LoRA：

1. **NF4 量化**: 将基础权重量化为 4-bit Normal Float
2. **Block-wise Scaling**: 每 64 个元素一个 absmax scale
3. **Full-Precision Adapters**: LoRA A/B 矩阵保持 fp16/bf16

```python
# 内存对比 (7B 模型)
Full FT:  14GB (fp16) + 14GB (grads) = 28GB
LoRA:     14GB (fp16) + 0.1GB = 14.1GB
QLoRA:    3.5GB (4-bit) + 0.1GB = 3.6GB  # ~4x 减少
```

## Consequences

**优势**:

- 显存占用降低 ~4x
- 可在 8GB GPU 上微调 7B 模型
- 与标准 LoRA 兼容的 API

**劣势**:

- 前向传播需要反量化，略有性能开销
- 无法像 LoRA 一样合并权重（需保持量化状态）
- 精度略有损失（NF4 是有损压缩）

**替代方案**: 对于推理场景，应使用标准 LoRA 并调用 `merge_lora()` 消除推理开销。
