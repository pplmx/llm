# 预训练教程

本教程介绍如何使用 `train_simple_decoder.py` 脚本从头训练一个小型语言模型。

## 快速开始

### 1. 准备数据

创建一个纯文本文件，每行一个样本：

```bash
echo -e "hello world\nthis is a test\nmachine learning is fun" > train.txt
```

### 2. 运行训练

```bash
uv run scripts/train_simple_decoder.py --file-path train.txt --epochs 3
```

### 3. 查看结果

训练完成后，会显示：
- 困惑度 (Perplexity)
- 损失值 (Loss)
- 学习率

---

## 完整示例

### 基本训练

```bash
# 使用 GPU 训练
uv run scripts/train_simple_decoder.py \
    --file-path data/train.txt \
    --epochs 10 \
    --batch-size 32 \
    --hidden-size 256 \
    --num-layers 4 \
    --num-heads 4
```

### 使用验证集

```bash
uv run scripts/train_simple_decoder.py \
    --file-path data/train.txt \
    --val-file-path data/val.txt \
    --epochs 5 \
    --batch-size 32
```

### 使用 Checkpoint (推荐)

```bash
# 1. 训练并保存 checkpoint
uv run scripts/train_simple_decoder.py \
    --file-path data/train.txt \
    --save-dir ./checkpoints \
    --save-interval 50 \
    --epochs 3

# 2. 从 checkpoint 恢复训练
uv run scripts/train_simple_decoder.py \
    --file-path data/train.txt \
    --resume ./checkpoints/latest.pt \
    --epochs 5
```

---

## 参数说明

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--hidden-size` | 64 | 隐藏层维度 |
| `--num-layers` | 2 | Transformer 层数 |
| `--num-heads` | 2 | 注意力头数 |
| `--max-seq-len` | 32 | 最大序列长度 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch-size` | 16 | 批次大小 |
| `--epochs` | 1 | 训练轮数 |
| `--lr` | 0.001 | 学习率 |
| `--device` | auto | 设备 (cpu/cuda) |

### Checkpoint 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--save-dir` | ./checkpoints | 保存目录 |
| `--save-interval` | 100 | 保存间隔 (步) |
| `--resume` | None | 恢复的 checkpoint |

---

## 数据格式

### 纯文本格式

```txt
hello world
this is a sample text
machine learning is fun
```

### 推荐数据量

| 场景 | 建议数据量 |
|------|-----------|
| 快速测试 | 1KB - 100KB |
| 学习演示 | 1MB - 10MB |
| 实际训练 | 100MB+ |

---

## 训练技巧

### 1. 模型大小选择

| 数据量 | 建议配置 |
|--------|---------|
| < 1MB | hidden=64, layers=2, heads=2 |
| 1-10MB | hidden=128, layers=4, heads=4 |
| 10MB+ | hidden=256, layers=6, heads=8 |

### 2. 学习率

- 初始学习率：1e-3 到 1e-4
- 使用 cosine annealing (已内置)
- 数据量小建议用更小的学习率

### 3. Checkpoint 使用

- 训练大模型时，较小 `--save-interval` (如 50-100)
- 中断后可用 `--resume` 恢复
- `latest.pt` 始终指向最新 checkpoint

---

## 故障排除

### GPU 不可用

```bash
# 强制使用 CPU
uv run scripts/train_simple_decoder.py --file-path data.txt --device cpu
```

### CUDA 内存不足

```bash
# 减小 batch size
uv run scripts/train_simple_decoder.py --file-path data.txt --batch-size 8
```

### 训练 loss 不下降

- 检查数据质量
- 尝试大学习率
- 增加模型容量

---

## 下一步

训练完成后，可以：

1. **推理测试** - 使用 `generate()` 函数生成文本
2. **微调** - 使用 LoRA/QLoRA 进行微调
3. **导出部署** - 导出模型用于生产环境

详见 [Fine-Tuning Guide](guide-finetuning.md) 和 [Inference Guide](guide-inference.md)。