# 预训练教程

从零开始训练一个小型的语言模型，覆盖**主流路径** (`llm-train stream_lm`)。教程里的所有命令都和 `llm-train` CLI、YAML 配置、预置数据源 (`data_source`) 对齐——而不是单独的 demo 脚本。

---

## 概述

本教程涵盖：

- 数据准备（流式 + 离线两种）
- 主流路径：使用 `llm-train stream_lm` + YAML 配置
- 检查点管理（包含 **streaming data cursor** 续训）
- 切换到 SFT / DPO 的入口
- 训练监控

> **前提**：克隆仓库后 `uv sync --group streaming`（C4 / The Pile 等 HuggingFace 流式数据集需要 `datasets` 包）。

---

## 1. 数据准备

### 1.1 离线小语料（快速冒烟）

最简形式：一个纯文本文件，**每行一条样本**。

```bash
mkdir -p data
python -c "print('\n'.join(f'this is sample line {i}' for i in range(200)))" \
  > data/demo.txt
```

这种格式直接被 `LocalLineTextSource` 读取（`data_source: local`）。

### 1.2 真实数据集预设（生产规模）

C4 / The Pile / RedPajama 通过 `src/llm/data/presets.py` 提供**预设 triple**（`dataset_name` / `dataset_config` / `text_column`），不需要手查 HuggingFace ID：

```python
from llm.data.presets import C4_PRESET, THEPILE_PRESET, REDPAJAMA_PRESETS
from llm.training.core.config import DataConfig
from llm.data.presets import apply_to_config

cfg = DataConfig(data_source="hf")
apply_to_config(cfg, C4_PRESET)
# cfg.dataset_name == "allenai/c4"
# cfg.dataset_config == "en"
```

预设的好处：
- 用户只记 `c4` / `pile` / `redpajama/arxiv`，不用记 HuggingFace ID
- 切换数据集改一行配置即可
- 同一个 `dataset_name` 在不同环境有一致的语义

### 1.3 数据量建议

| 场景 | 建议数据量 | 数据源 |
| --- | --- | --- |
| 快速冒烟 | 1KB - 100KB | `data_source: local` + 文本文件 |
| 学习演示 | 1MB - 10MB | `data_source: hf` + 切小流的 `c4` 子集 |
| 实际训练 | 100GB+ | `data_source: hf` + C4 / The Pile / RedPajama |

---

## 2. 主流路径：`llm-train stream_lm`

**`stream_lm` task** 把 `LanguageModelingTask` 接到 `StreamingTextDataModule`，是流式预训练的主入口。

### 2.1 用预设配置直接跑

仓库自带三个 streaming 预设配置：

| 配置文件 | 用途 |
| --- | --- |
| `configs/streaming_local_demo.yaml` | 离线小语料冒烟（CPU 也能跑，几秒完成） |
| `configs/streaming_c4.yaml` | C4 'en' 子集，768 hidden × 12 layers，AMP + gradient checkpointing |
| `configs/example.yaml` | 非流式 LM baseline（参考） |

```bash
# 离线冒烟（推荐先跑这个验证环境）
uv run llm-train stream_lm --config configs/streaming_local_demo.yaml

# C4 真实流式（需要 GPU + datasets 包）
uv run llm-train stream_lm --config configs/streaming_c4.yaml
```

### 2.2 CLI 覆盖

`llm-train` 支持在命令行覆盖 YAML 字段，常用于实验 sweep：

```bash
uv run llm-train stream_lm --config configs/streaming_local_demo.yaml \
  --epochs 3 \
  --steps-per-epoch 20 \
  --batch-size 8 \
  --lr 5e-4 \
  --num-samples 5000
```

可覆盖字段：`--epochs`、`--batch-size`、`--lr`、`--num-samples`、`--steps-per-epoch`、`--compile`（默认 True）、`--amp`（默认 True）。

### 2.3 直接写 YAML（推荐生产用）

`configs/streaming_local_demo.yaml` 的结构（精简版）：

```yaml
model:
  hidden_size: 64
  num_heads: 4
  num_layers: 2
  vocab_size: 256          # SimpleCharacterTokenizer 用小 vocab
  max_seq_len: 64

training:
  epochs: 1
  batch_size: 4
  learning_rate: 1.0e-3
  num_samples: 1000

data:
  data_source: local
  dataset_path: data/demo.txt
  max_seq_len: 64
  steps_per_epoch: 10       # 流式数据无固定 epoch 长度 → 用户指定步数

optimization:
  use_amp: false            # CPU 冒烟 → 关闭 AMP
  use_compile: false        # 冒烟 → 关闭 torch.compile（节省 warmup）

checkpoint:
  checkpoint_dir: checkpoints_streaming_demo
  save_interval: 1
```

**`data_source` 的合法值**（详见 `DataConfig` 正则）：

- `local` —— 本地文件，按行读取
- `hf` —— HuggingFace 流式（需要 `dataset_name` + `dataset_config`）
- `dedup_local` / `dedup_hf` —— T3 #39 dedup 包装层（跨 run 去重）

### 2.4 自定义模型规模

隐藏维度、层数、注意力头数都通过 `model:` 段配置。生产推荐先用 `streaming_c4.yaml` 的 768 × 12，再按显存扩展：

```yaml
model:
  hidden_size: 1024        # C4 起步 768；1024 / 2048 更常见
  num_layers: 24
  num_heads: 16
  num_kv_heads: 4          # GQA：KV 头比 Q 头少，省 KV cache
  intermediate_size: 4096  # 默认 4*hidden；可手动调
  vocab_size: 50257        # GPT-2 BPE
  max_seq_len: 2048        # 长上下文要 gradient_checkpointing
```

---

## 3. Checkpoint 管理（流式 cursor-aware）

### 3.1 自动保存

流式训练和普通 LM 的 checkpoint **结构相同**，但额外在 `extra_state["stream_data"]` 里存**每个 shard 的 line_index**（每条流式数据源被读到哪里）：

```text
checkpoints/
├── latest.pt
├── epoch_1.pt
├── epoch_2.pt
└── best.pt
```

`epoch_N.pt` 里包含：

- `model_state`：模型权重（adapter 部分也包含在内，详见 [PEFT save/load](../reference/architecture.md#peft-method-registry)）
- `optimizer_state`：AdamW 状态
- `scheduler_state`：LR scheduler
- `scaler_state`：AMP GradScaler
- `extra_state["stream_data"]`：`{shard_key: {line_index, token_buffer}}`
- `extra_state["stream_source"]`：数据源指纹（防止 config drift）

### 3.2 Resume（含 cursor）

```bash
# 第一次跑（生成 checkpoints/epoch_1.pt）
uv run llm-train stream_lm --config configs/streaming_local_demo.yaml --epochs 2

# 接着再跑 2 个 epoch（cursor 自动续上）
uv run llm-train stream_lm --config configs/streaming_local_demo.yaml \
  --resume-from-checkpoint checkpoints/epoch_2.pt
```

**Resume 保证**：
- `model_state` / `optimizer_state` / `scheduler_state` 全部还原
- 数据 cursor 接着上次的 `line_index`，不会重复读 / 漏读
- 如果你换了 `data_source` / `dataset_name`，resume 会报警（`stream_source` 指纹不一致）

### 3.3 检查 checkpoint 内容

```python
import torch

ckpt = torch.load("checkpoints/epoch_1.pt", map_location="cpu", weights_only=False)
print("epoch:", ckpt["epoch"])
print("loss:", ckpt["loss"])
print("stream cursor:", ckpt["extra_state"]["stream_data"])
# → {"0": {"line_index": 12, "token_buffer": [...]}}
```

---

## 4. 切换到 SFT / DPO

`llm-train` 注册了多个 task，**PEFT + 流式训练同样适用**：

```bash
# 监督微调（instruction tuning）
uv run llm-train sft --config configs/your_sft_config.yaml

# 直接偏好优化（DPO）
uv run llm-train dpo --config configs/your_dpo_config.yaml
```

PEFT（LoRA / IA³ / BitFit / Adapter / Prefix Tuning / AdaLoRA）通过 `training.peft_method` + `training.peft_kwargs` 启用——同一份 YAML 格式，无侵入：

```yaml
training:
  peft_method: lora
  peft_kwargs:
    rank: 8
    alpha: 16.0
  peft_save_path: checkpoints/lora_adapter.bin   # 训练结束自动写一份
```

详见 [微调教程](./02-finetuning.md)。

---

## 5. 训练监控

### 5.1 日志格式

每个 epoch 总结一行：

```text
Epoch  1/1 SUMMARY | Train Loss: 4.6700 | LR: 0.001000 | Time: 7.14s | Peak Mem: 0.00 GB
```

- `Train Loss`：cross-entropy（label-smoothed 视配置而定）
- `LR`：当前学习率（warmup + cosine schedule）
- `Time`：epoch 耗时
- `Peak Mem`：GPU 峰值显存（CPU 训练为 0）

### 5.2 TensorBoard

```yaml
logging:
  log_dir: runs/streaming_demo
```

```bash
uv run tensorboard --logdir runs/
```

PEFT 训练还会写额外指标：`adalora/effective_rank`（T3 #42）。

### 5.3 验证集

```yaml
data:
  val_dataset_path: data/val.txt   # 路径可以不同于训练集
```

未设置 → 跳过验证（节省时间）。流式验证集也是按行读取。

---

## 6. 故障排除

### 6.1 CUDA OOM

```yaml
# 减小 batch size 或开 gradient checkpointing
training:
  batch_size: 4             # 减小
optimization:
  gradient_checkpointing: true
data:
  max_seq_len: 512          # 缩短
```

### 6.2 Streaming 报 "data cursor mismatch"

Resume 时如果 `stream_source` 指纹（hash of data config）和 checkpoint 里的不一致，**loud failure**：

```text
RuntimeError: stream source fingerprint mismatch — config drifted since checkpoint
```

这是设计：避免"看起来在恢复，实际上读的是不同数据"的隐蔽 bug。改回原 config 或重新跑。

### 6.3 Tokenizer 不匹配

`SimpleCharacterTokenizer` 是按**构造时的 corpus** 锁定 vocab 的。如果你的训练数据包含 corpus 里没出现过的字符：

```text
KeyError: "Character 'X' not found in tokenizer vocabulary"
```

解决：
- 生产用 `tokenizer_type: hf` + `tokenizer_path: gpt2`（BPE，无字符级 vocab 问题）
- 或者用 `tests/support/tokenizers.py:LineTokenizer`（按行 tokenize）

### 6.4 检查点损坏

检查点写入是**原子操作**（先写 `.tmp` 再 `rename`），损坏几乎只会因为：
- 磁盘满
- 进程被 kill 在 `.tmp` 阶段（`.tmp` 文件残留，可删）

---

## 7. 进阶：从教程到生产

教程用的 `streaming_local_demo.yaml` 是**冒烟级**。生产规模通常要：

1. **数据**：C4 / The Pile / RedPajama 的 100GB+ 子集；启用 `dedup_local` / `dedup_hf` 跨 run 去重
2. **模型**：hidden_size 768-2048，num_layers 12-48，配合 GQA / SwiGLU / MoE
3. **优化**：AMP + gradient checkpointing + torch.compile（`mode: reduce-overhead` 或 `max-autotune`）
4. **并行**：`parallel_strategy: ddp`（多 GPU）或 `fsdp`（多节点）
5. **PEFT**：大模型 + 小数据时 `peft_method: lora`，训练产物只是 `lora_adapter.bin`（几 MB），可分发到不同 base 上

详见：
- [架构文档](../reference/architecture.md)
- [分布式训练](../guides/distributed.md)
- [PEFT 微调](../guides/finetuning.md)

---

## 下一步

| 目标 | 文档 |
| --- | --- |
| 多 GPU / 多节点训练 | [Guides/分布式训练](../guides/distributed.md) |
| LoRA / QLoRA 微调 | [Tutorials/微调](./02-finetuning.md) |
| 推理优化（Paged Attention / Continuous Batching） | [Guides/推理](../guides/inference.md) |
| 模型评估（lm-eval-harness / Perplexity） | [Guides/评估](../guides/evaluation.md) |
