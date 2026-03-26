# Checkpoint 完整指南

训练过程中的模型保存与恢复。

## 概述

Checkpoint 包含：
- 模型权重
- 优化器状态
- 训练进度 (epoch, step)
- 训练配置

---

## 使用方式

### 保存 Checkpoint

```bash
# 训练时自动保存
uv run scripts/train_simple_decoder.py \
    --file-path data.txt \
    --save-dir ./checkpoints \
    --save-interval 100
```

保存位置：
```
checkpoints/
├── checkpoint_step_100.pt
├── checkpoint_step_200.pt
└── latest.pt  (最新)
```

### 恢复训练

```bash
# 从 latest 恢复
uv run scripts/train_simple_decoder.py \
    --file-path data.txt \
    --resume ./checkpoints/latest.pt \
    --epochs 5

# 从指定 checkpoint 恢复
uv run scripts/train_simple_decoder.py \
    --file-path data.txt \
    --resume ./checkpoints/checkpoint_step_500.pt
```

---

## Checkpoint 结构

```python
{
    "model_state_dict": {...},      # 模型权重
    "optimizer_state_dict": {...},  # 优化器状态
    "epoch": 2,                    # 当前 epoch
    "global_step": 500,            # 总步数
    "loss": 1.234,                # 当前 loss
    "config": {                    # 训练配置
        "hidden_size": 256,
        "num_layers": 4,
        ...
    }
}
```

---

## 手动保存/加载

### 保存

```python
import torch

checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
    "global_step": global_step,
    "loss": loss,
    "config": {...}
}

torch.save(checkpoint, "model.pt")
```

### 加载

```python
checkpoint = torch.load("model.pt")

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

epoch = checkpoint["epoch"]
global_step = checkpoint["global_step"]
```

---

## 最佳实践

### 1. 定期保存

```bash
# 较小间隔保存 (数据大时)
--save-interval 50
```

### 2. 保留最佳

可以比较 loss 选择最佳 checkpoint：

```python
best_loss = float('inf')
best_ckpt = None
for ckpt in checkpoints:
    if ckpt["loss"] < best_loss:
        best_loss = ckpt["loss"]
        best_ckpt = ckpt
```

### 3. 清理旧 Checkpoint

```bash
# 保留最新的 N 个
ls -t checkpoints/*.pt | tail -n +5 | xargs rm
```

---

## 故障排除

**Q: Checkpoint 损坏？**
- 检查文件完整性 `ls -la checkpoint.pt`
- 尝试加载看是否报错

**Q: 模型不匹配？**
- 确保 checkpoint 的 config 与当前模型配置一致
- 检查 hidden_size, num_layers 等

**Q: 训练中断？**
- 使用 `--resume` 从 latest.pt 恢复
- 确保 --save-dir 路径正确

---

## 相关文档

- [预训练教程](../tutorials/01-pretraining.md)
- [推理服务](../tutorials/03-inference.md)