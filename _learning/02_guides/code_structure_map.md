# 项目代码结构指南

这份指南将帮助你将 `_learning/docs/roadmap.md` 中的理论知识与 `src/llm` 中的实际代码对应起来。

## 核心模块 (`src/llm/core`)

这是学习 Transformer 架构最核心的部分。

| 模块文件 | 对应概念 | 说明 |
|---------|----------|------|
| `attn/mha.py` | Multi-Head Attention, GQA, MQA | 实现了标准多头注意力和分组查询注意力(GQA) |
| `rope.py` | Rotary Position Embedding (RoPE) | 如果你正在学习位置编码，这里展示了如何实现旋转编码 |
| `alibi.py` | ALiBi | 另一种强大的位置编码方案，通过线性偏置实现 |
| `mlp.py` | Feed Forward Network (FFN) | Transformer 的前馈网络层，支持 SwiGLU 激活函数 |
| `moe/` | Mixture of Experts (MoE) | 如果你想了解稀疏混合专家模型，请查看此目录 |
| `layer_norm.py` | Layer Normalization | 标准层归一化实现 |
| `rms_norm.py` | RMSNorm | LLaMA 等模型使用的归一化变体 |
| `transformer_block.py` | Transformer Layer | 将 Attention 和 MLP 组合成一个完整的 Encoder/Decoder 层 |

## 模型架构 (`src/llm/model`)

这里展示了如何将核心组件组装成完整的语言模型。

- `decoder_model.py`: 类似于 GPT/LLaMA 的 Decoder-only架构实现。它是理解现代 LLM 如何工作的最佳切入点。

## 使用建议

1. **阅读源码**: 不要只看文档，直接阅读代码是学习深度学习架构最快的方式。关键逻辑通常只有几十行。
2. **运行测试**: `tests/` 目录下的测试用例展示了如何使用这些模块。尝试运行特定的测试来观察输入输出形状。
   - `tests/core/test_attention.py`: 观察 Attention 的 QKV 变换
   - `tests/core/test_rope.py`: 观察位置编码如何作用于 Tensor
3. **断点调试**: 在测试中设置断点，单步执行，观察 Tensor 维度的变化。

## 常见问题

- **Q: GQA 是在哪里实现的？**
  A: 在 `src/llm/core/attn/mha.py` 中。注意看 `num_kv_heads` 参数，当它小于 `num_heads` 时就是 GQA。

- **Q: SwiGLU 激活函数在哪里？**
  A: 在 `src/llm/core/mlp.py` 的 `MLP` 类中，通过 `activation="swiglu"` 启用。
