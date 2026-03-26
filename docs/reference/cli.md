# CLI 命令参考

所有命令行工具的完整参数列表。

---

## train_simple_decoder.py

预训练模型脚本。

### 位置

```bash
scripts/train_simple_decoder.py
```

### 参数

#### 必需参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `--file-path` | Path | 训练数据文件路径 |

#### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--hidden-size` | 64 | 隐藏层维度 |
| `--num-layers` | 2 | Transformer 层数 |
| `--num-heads` | 2 | 注意力头数 |
| `--max-seq-len` | 32 | 最大序列长度 |

#### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch-size` | 16 | 批次大小 |
| `--epochs` | 1 | 训练轮数 |
| `--lr` | 1e-3 | 学习率 |
| `--device` | auto | 设备 (cpu/cuda) |
| `--log-interval` | 10 | 日志输出间隔 |

#### Checkpoint 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--save-dir` | ./checkpoints | 保存目录 |
| `--save-interval` | 100 | 保存间隔 (步数) |
| `--resume` | None | 恢复的 checkpoint 路径 |

#### 早停参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--val-file-path` | None | 验证数据文件 |
| `--early-stopping-patience` | 3 | 早停耐心值 |
| `--early-stopping-min-delta` | 0.001 | 最小改善阈值 |

### 使用示例

```bash
# 基本使用
python scripts/train_simple_decoder.py --file-path data.txt

# 带 checkpoint
python scripts/train_simple_decoder.py \
    --file-path data.txt \
    --save-dir ./checkpoints \
    --save-interval 50

# 从 checkpoint 恢复
python scripts/train_simple_decoder.py \
    --file-path data.txt \
    --resume ./checkpoints/latest.pt
```

---

## 训练配置 (YAML)

使用配置文件进行复杂训练：

```yaml
model:
  hidden_size: 256
  num_layers: 4
  num_heads: 4
  vocab_size: 32000

training:
  batch_size: 32
  epochs: 3
  lr: 1e-4

optimization:
  use_amp: true
  use_compile: true

distributed:
  backend: nccl
  world_size: 4
```

---

## 环境变量

| 变量 | 说明 |
|------|------|
| `CUDA_VISIBLE_DEVICES` | 可见 GPU 设备 |
| `NCCL_DEBUG` | NCCL 调试级别 |
| `TORCH_CUDNN_V8_ENABLED` | cuDNN v8 优化 |