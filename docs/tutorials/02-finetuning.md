# 微调教程

在预训练模型基础上进行微调。

## 概述

| 方法       | 显存占用 | 适用场景              |
| ---------- | -------- | --------------------- |
| 全参数微调 | 100%     | 小模型、显存充足      |
| LoRA       | ~10%     | 通用 PEFT，平衡效果好 |
| QLoRA      | ~5%      | 大模型、显存有限      |

---

## 快速开始

### 1. 准备数据

对话格式数据 (SFT)：

```json
{"instruction": "什么是机器学习？", "input": "", "output": "机器学习是..."}
{"instruction": "写一首诗", "input": "春天", "output": "春风拂面..."}
```

### 2. 使用 LoRA 微调

```python
import torch
from llm.models.decoder import DecoderModel
from llm.core.lora import apply_lora, get_lora_parameters, merge_lora
from llm.training.tasks.sft_task import SFTTask

# 1. 加载预训练模型
model = DecoderModel(
    vocab_size=32000,
    hidden_size=768,
    num_layers=12,
)

# 2. 应用 LoRA
apply_lora(
    model,
    rank=8,
    alpha=16.0,
    target_modules=["qkv_proj", "out_proj"],
)

# 3. 只优化 LoRA 参数
optimizer = torch.optim.AdamW(
    get_lora_parameters(model),
    lr=1e-4
)

# 4. 训练 (使用 SFTTask)
task = SFTTask(config, data_module)
for batch in dataloader:
    loss = task.train_step(batch, model, criterion)
    loss.backward()
    optimizer.step()

# 5. 推理时合并权重
merge_lora(model)
```

---

## LoRA 参数选择

| 场景     | rank | alpha | 说明         |
| -------- | ---- | ----- | ------------ |
| 快速测试 | 4    | 8     | 最少参数     |
| 一般任务 | 8    | 16    | 推荐默认值   |
| 复杂任务 | 16   | 32    | 更多表达能力 |

---

## QLoRA (更少显存)

```python
from llm.core.qlora import apply_qlora, get_qlora_parameters

# 使用 4-bit 量化
apply_qlora(
    model,
    rank=8,
    nf4_quantization=True,
)

optimizer = torch.optim.AdamW(
    get_qlora_parameters(model),
    lr=1e-4
)
```

---

## 详细文档

完整的微调指南请参考：

- [Fine-Tuning Guide](../guides/finetuning.md) - 详细参数说明
- [QLoRA Guide](../guides/finetuning.md#qlora-quantized-lora) - 量化微调

---

## 常见问题

**Q: LoRA 效果不好？**

- 尝试增大 rank (8 → 16)
- 调整 target_modules
- 检查数据质量

**Q: 显存不够？**

- 使用 QLoRA
- 减小 batch size
- 使用 gradient accumulation

---

## 下一步

| 目标         | 文档                                          |
| ------------ | --------------------------------------------- |
| 部署推理服务 | [Tutorials/推理服务](./03-inference.md)       |
| 分布式训练   | [Guides/分布式训练](../guides/distributed.md) |
