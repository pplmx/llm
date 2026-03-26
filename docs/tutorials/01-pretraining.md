# 预训练教程

从零开始训练一个小型语言模型。

## 概述

本教程涵盖：
- 数据准备
- 模型训练
- Checkpoint 管理
- 训练监控

---

## 1. 数据准备

### 数据格式

纯文本文件，每行一个样本：

```text
hello world
this is a sample text
machine learning is fun and powerful
natural language processing enables computers to understand text
```

### 数据量建议

| 场景 | 建议数据量 |
|------|-----------|
| 快速测试 | 1KB - 100KB |
| 学习演示 | 1MB - 10MB |
| 实际训练 | 100MB+ |

### 数据预处理

```python
# 可以使用训练脚本内置的 tokenizer
# 会自动从文本构建词汇表
```

---

## 2. 基础训练

```bash
uv run scripts/train_simple_decoder.py \
    --file-path data/train.txt \
    --epochs 3 \
    --batch-size 32 \
    --hidden-size 256 \
    --num-layers 4 \
    --num-heads 4
```

### 参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--file-path` | 训练数据文件 | 必需 |
| `--epochs` | 训练轮数 | 3-10 |
| `--batch-size` | 批次大小 | 16-64 |
| `--hidden-size` | 隐藏层维度 | 128-512 |
| `--num-layers` | Transformer 层数 | 2-8 |
| `--num-heads` | 注意力头数 | 2-8 |
| `--lr` | 学习率 | 1e-4 到 1e-3 |

---

## 3. Checkpoint 管理

### 自动保存

```bash
# 每100步自动保存
uv run scripts/train_simple_decoder.py \
    --file-path data/train.txt \
    --save-dir ./checkpoints \
    --save-interval 100
```

生成的文件：
```
checkpoints/
├── checkpoint_step_100.pt
├── checkpoint_step_200.pt
├── checkpoint_step_300.pt
└── latest.pt  (最新)
```

### 从 Checkpoint 恢复

```bash
uv run scripts/train_simple_decoder.py \
    --file-path data/train.txt \
    --resume ./checkpoints/latest.pt \
    --epochs 5
```

### Checkpoint 内容

```python
{
    "model_state_dict": {...},
    "optimizer_state_dict": {...},
    "epoch": 2,
    "global_step": 500,
    "loss": 1.234,
    "config": {
        "hidden_size": 256,
        "num_layers": 4,
        ...
    }
}
```

---

## 4. 训练监控

### 进度条

训练时显示实时进度：
```
Epoch 1/3: 100%|██████████| 100/100 [00:05<00:00, loss=2.34, step=100]
```

### 损失值

每个 epoch 结束后显示：
- 平均训练损失
- 学习率

### 验证集

```bash
uv run scripts/train_simple_decoder.py \
    --file-path data/train.txt \
    --val-file-path data/val.txt \
    --epochs 3
```

---

## 5. 完整示例

### 完整训练流程

```bash
# 1. 准备数据
mkdir -p data
echo "your training data" > data/train.txt

# 2. 训练 (带 checkpoint)
uv run scripts/train_simple_decoder.py \
    --file-path data/train.txt \
    --val-file-path data/val.txt \
    --save-dir ./checkpoints \
    --save-interval 50 \
    --epochs 10 \
    --batch-size 32 \
    --hidden-size 256 \
    --num-layers 4 \
    --num-heads 4 \
    --lr 1e-4

# 3. 查看结果
ls -la checkpoints/

# 4. 从最佳 checkpoint 继续训练
uv run scripts/train_simple_decoder.py \
    --file-path data/train.txt \
    --resume ./checkpoints/latest.pt \
    --epochs 5
```

---

## 6. 故障排除

### GPU 内存不足

```bash
# 减小 batch size
--batch-size 8

# 使用 CPU
--device cpu

# 减小模型
--hidden-size 128 --num-layers 2
```

### 训练不收敛

- 检查数据质量
- 调整学习率 (尝试 1e-4)
- 增加模型容量

### Checkpoint 损坏

重新训练，checkpoint 只保存模型权重，如果文件损坏需要从头训练。

---

## 下一步

| 目标 | 文档 |
|------|------|
| 使用多 GPU 训练 | [Guides/分布式训练](guides/distributed.md) |
| LoRA 微调 | [Tutorials/微调](tutorials/02-finetuning.md) |
| 优化推理速度 | [Guides/推理优化](guides/optimization.md) |