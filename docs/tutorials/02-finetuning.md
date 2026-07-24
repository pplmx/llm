# 微调教程

在预训练模型基础上做**监督微调 (SFT)** 与**直接偏好优化 (DPO)** 的主路径教程。和 `01-pretraining.md` 一样，所有命令都对齐 `llm-train` CLI + YAML 配置——而不是单独的 demo 脚本。

---

## 概述

本教程涵盖：

- 数据准备（Alpaca / Dolly / OASST 风格的 JSONL）
- 主流路径：`llm-train sft` + `llm-train dpo` + YAML
- 任务差异：SFT 是单模型交叉熵；DPO 是 policy + reference 双模型的偏好损失
- 检查点管理
- **PEFT 集成 + 服务化**（LoRA / IA³ / BitFit / Adapter / Pfeiffer / AdaLoRA → `llm-serve` 一条龙）
- 训练监控

> **前提**：上游已经有预训练 checkpoint（参考 [01-pretraining.md](./01-pretraining.md)）。SFT / DPO 通常是预训练之后的第二步。

---

## 1. 数据准备

### 1.1 SFT 格式：Alpaca JSONL

每行一个样本，三段式：

```json
{"instruction": "什么是机器学习？", "input": "", "output": "机器学习是..."}
{"instruction": "写一首诗", "input": "春天", "output": "春风拂面..."}
```

字段含义：

- `instruction` — 任务描述（必填）
- `input` — 可选的额外上下文
- `output` — 期望的响应（必填）

`SFTDataset.alpaca_template` 自动把 `instruction` / `input` 渲染成 prompt，`output` 作为 response，response 之前的 token 在 loss 中被 mask 掉（用 `ignore_index=-100`），只对 response 计算损失。

### 1.2 DPO 格式：chosen / rejected JSONL

DPO 用偏好对（preference pair），每行三个字段：

```json
{"prompt": "什么是机器学习？", "chosen": "机器学习是让计算机从数据中自动学习规律。", "rejected": "我不知道。"}
{"prompt": "写一首诗", "chosen": "春风拂面杨柳青，碧波荡漾画中行。", "rejected": "随便写写就好。"}
```

字段含义：

- `prompt` — 共享的前缀（chosen 和 rejected 共用）
- `chosen` — 人类偏好的响应
- `rejected` — 人类不偏好的响应

DPO 的目标是让 policy 模型给 chosen 的概率高于 rejected，同时不要太远离 reference 模型（KL 约束由 `dpo_beta` 控制）。

### 1.3 离线小语料（快速冒烟）

最简形式：几行 JSONL 就能跑通。

```bash
mkdir -p data
python -c "
import json

# SFT 冒烟数据（Alpaca 风格）
sft_data = [
    {'instruction': 'Greet me.', 'input': '', 'output': 'Hello!'},
    {'instruction': 'What is 2+2?', 'input': '', 'output': '4'},
    {'instruction': 'Translate to French.', 'input': 'hi', 'output': 'salut'},
] * 30
with open('data/sft_demo.jsonl', 'w') as f:
    for item in sft_data:
        f.write(json.dumps(item) + '\n')

# DPO 冒烟数据（chosen/rejected）
dpo_data = [
    {'prompt': 'What is 2+2?', 'chosen': '4', 'rejected': '5'},
    {'prompt': 'Capital of France?', 'chosen': 'Paris', 'rejected': 'London'},
] * 30
with open('data/dpo_demo.jsonl', 'w') as f:
    for item in dpo_data:
        f.write(json.dumps(item) + '\n')
"
```

### 1.4 真实数据集

#### SFT（Alpaca）

```python
# Alpaca 是 52K 行 CSV，需要转成 JSONL：
import csv, json

with open("data/alpaca.jsonl", "w") as out:
    for row in csv.DictReader(open("alpaca_data.csv")):
        out.write(json.dumps(row) + "\n")
```

转换后的 JSONL 字段名 (`instruction` / `input` / `output`) 和 `SFTDataset.alpaca_template` 完全对齐，无需额外配置。

#### DPO（UltraFeedback）

```python
from datasets import load_dataset
import json

ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
with open("data/ultrafeedback.jsonl", "w") as out:
    for row in ds:
        out.write(
            json.dumps(
                {
                    "prompt": row["prompt"],
                    "chosen": row["chosen"][1]["content"],
                    "rejected": row["rejected"][1]["content"],
                }
            )
            + "\n"
        )
```

转换后的字段 (`prompt` / `chosen` / `rejected`) 直接匹配 `DPODataset`。

### 1.5 数据量建议

| 场景 | 建议数据量 | 数据源 |
| --- | --- | --- |
| 快速冒烟 | 10-100 条 | 手写 JSONL |
| 学习演示 | 1K-10K 条 | 切小流的 Alpaca / Dolly 子集 |
| 实际 SFT | 50K-500K 条 | Alpaca (52K) / Dolly (15K) / OASST |
| 实际 DPO | 50K-100K 对 | UltraFeedback / Anthropic hh-rlhf |

---

## 2. 主流路径：`llm-train sft`

### 2.1 用预设配置直接跑

仓库自带两个 SFT 预设：

| 配置文件 | 用途 |
| --- | --- |
| `configs/sft_local_demo.yaml` | 离线 JSONL 冒烟（CPU 也能跑，几秒完成） |
| `configs/sft_alpaca.yaml` | 生产 Alpaca JSONL（256 hidden × 6 layers，AMP） |

```bash
# 离线冒烟（推荐先跑这个验证环境）
uv run llm-train sft --config configs/sft_local_demo.yaml

# Alpaca 真实 SFT（需要 GPU + GPT-2 tokenizer）
uv run llm-train sft --config configs/sft_alpaca.yaml
```

### 2.2 CLI 覆盖

```bash
uv run llm-train sft --config configs/sft_local_demo.yaml \
  --epochs 3 \
  --steps-per-epoch 20 \
  --batch-size 8 \
  --lr 5e-4 \
  --num-samples 5000
```

可覆盖字段：`--epochs`、`--batch-size`、`--lr`、`--num-samples`、`--steps-per-epoch`、`--compile`（默认 True）、`--amp`（默认 True）。`--peft-method` / `--peft-kwargs` / `--peft-save-path` 在新版 CLI 也可覆盖（详见 §5）。

### 2.3 YAML 结构（推荐生产用）

`configs/sft_local_demo.yaml` 的结构（精简版）：

```yaml
model:
  hidden_size: 64          # tiny for the demo; production: 512-4096
  num_heads: 4
  num_layers: 2
  vocab_size: 256          # SimpleCharacterTokenizer 用小 vocab
  max_seq_len: 64

training:
  epochs: 1
  batch_size: 4
  learning_rate: 1.0e-3
  num_samples: 1000
  max_steps: 50
  warmup_epochs: 0

data:
  tokenizer_type: simple
  dataset_path: data/sft_demo.jsonl   # Alpaca JSONL
  max_seq_len: 64

optimization:
  use_amp: false           # CPU demo
  use_compile: false       # 冒烟 → 关闭 torch.compile

checkpoint:
  checkpoint_dir: checkpoints_sft_demo
  save_interval: 1
```

### 2.4 SFT 的 train_step

`SFTTask.train_step` 接收一个 `dict[str, Tensor]`：

- `input_ids`：prompt + response 的 token ids
- `labels`：prompt 部分是 `-100`（mask），response 部分是真实 token
- `attention_mask`：可选，1 表示真实 token，0 表示 padding

**关键设计**：loss 只在 response 部分计算（SFT 是教模型生成 response，而不是复读 prompt），通过把 prompt 位置的 `labels` 设成 `ignore_index=-100` 实现。

### 2.5 Resume

```bash
# 第一次跑（生成 checkpoints/epoch_1.pt）
uv run llm-train sft --config configs/sft_local_demo.yaml --epochs 2

# 接着再跑 2 个 epoch
uv run llm-train sft --config configs/sft_local_demo.yaml \
  --resume-from-checkpoint checkpoints/epoch_2.pt
```

SFT 是 map-style dataset（已知 epoch 长度），不像流式预训练需要 cursor resume——`CheckpointManager.load_checkpoint` 恢复 `model_state` / `optimizer_state` / `scheduler_state` / `start_epoch` 就够了。

---

## 3. 主流路径：`llm-train dpo`

### 3.1 启动方式

```bash
uv run llm-train dpo --config configs/dpo_local_demo.yaml
```

`DPOTask.build_model` 一次性构造 **policy + reference** 两个模型：

- policy：可训练（`requires_grad=True`）
- reference：frozen 副本（`eval()` + `requires_grad=False`），从同一组初始权重 load_state_dict

### 3.2 关键差异（vs SFT）

| 维度 | SFT | DPO |
| --- | --- | --- |
| 数据 | `{instruction, output}` | `{prompt, chosen, rejected}` |
| 模型数 | 1（policy） | 2（policy + reference） |
| Loss | cross-entropy on response | log-ratio + KL penalty |
| 显存 | baseline | 约 2×（policy + ref） |
| 学习率 | 1e-5 ~ 5e-5 | 5e-7 ~ 1e-6（小一个量级） |
| 批次 | 16-64 | 4-16（小一半，gradient_accumulation 补） |
| beta | — | `training.dpo_beta`（默认 0.1） |

### 3.3 DPO 的 train_step

`DPOTask.train_step` 同时 forward chosen + rejected：

```text
policy_chosen_logps = log_softmax(policy(chosen_ids)) @ chosen_labels
policy_rejected_logps = log_softmax(policy(rejected_ids)) @ rejected_labels
with no_grad:
    ref_chosen_logps = log_softmax(ref(chosen_ids)) @ chosen_labels
    ref_rejected_logps = log_softmax(ref(rejected_ids)) @ rejected_labels

# DPO loss: log-sigmoid of beta * (policy_log_ratio - ref_log_ratio)
loss = -logsigmoid(beta * (
    (policy_chosen_logps - ref_chosen_logps) -
    (policy_rejected_logps - ref_rejected_logps)
)).mean()
```

### 3.4 SFT → DPO 的标准两阶段

```bash
# 阶段 1：SFT
uv run llm-train sft --config configs/sft_alpaca.yaml
# → checkpoints_sft_alpaca/epoch_3.pt

# 阶段 2：DPO 从 SFT ckpt 续训
# 编辑 configs/dpo_ultrafeedback.yaml，设置：
#   checkpoint.resume_from_checkpoint: checkpoints_sft_alpaca/epoch_3.pt
uv run llm-train dpo --config configs/dpo_ultrafeedback.yaml
```

DPO **必须从已微调的 policy 开始**（不能直接从 base 模型做偏好对齐——reference 和 policy 差异过大时 DPO loss 不稳定）。

### 3.5 `dpo_beta` 的影响

| beta | 行为 |
| --- | --- |
| 0.01-0.05 | 弱 KL 约束，policy 可以大幅偏离 reference；DPO loss 不稳定 |
| 0.1（默认） | 文献标准值，平衡稳定性与对齐强度 |
| 0.3-0.5 | 强 KL 约束，DPO 退化成"接近 SFT"的回归任务 |

---

## 4. PEFT 集成 + 服务化（T2 PEFT #47-#49）

SFT / DPO 都支持 PEFT——同一份 YAML 格式，无需额外 wiring：

```yaml
training:
  peft_method: lora
  peft_kwargs:
    rank: 8
    alpha: 16.0
  peft_save_path: checkpoints_sft_demo/lora_adapter.bin   # 训练结束自动写一份
```

`PEFTAdapterCheckpointCallback`（T2 PEFT #48）自动随 `build_callbacks()` 注册，`on_train_end` 调 `save_peft` 把 adapter sidecar 写到 `peft_save_path`（默认 `{checkpoint_dir}/peft_adapter_{method}.bin`）。**8 个内置方法全支持**：lora / qlora / adalora / ia3 / bitfit / adapter / pfeiffer_adapter / prefix_tuning。

### 4.1 训练 → 服务化一条龙

```bash
# 1. SFT + LoRA
uv run llm-train sft --config configs/sft_alpaca.yaml \
  --peft-method lora \
  --peft-kwargs '{"rank":16,"alpha":32.0}' \
  --peft-save-path ./lora_adapter.bin

# 2. 服务化（T2 PEFT #49）— 直接通过环境变量挂载 adapter
LLM_SERVING_MODEL_PATH=./checkpoints_sft_alpaca/epoch_3.pt \
LLM_SERVING_TOKENIZER_PATH=./tokenizer.pt \
LLM_SERVING_PEFT_METHOD=lora \
LLM_SERVING_PEFT_ADAPTER_PATH=./lora_adapter.bin \
LLM_SERVING_PEFT_KWARGS='{"rank":16,"alpha":32.0}' \
  uv run llm-serve
```

LoRA / IA³ / Adapter / Pfeiffer 可以加 `LLM_SERVING_PEFT_MERGE=true` 把 adapter fold 进 base weights（节省 per-token routing 开销，但失去 disable/unmerge 能力）。BitFit / QLoRA / Prefix Tuning 不支持 merge（配置 validator 在启动时拒绝 `peft_merge=true`）。

### 4.2 DPO 是否要 PEFT？

**经验法则**：

- 模型 < 1B：直接全参数 DPO，PEFT 收益小
- 模型 > 7B：**必须 PEFT**——reference 模型副本占显存太大
- DPO + LoRA：把 reference 模型也 LoRA 化（同一个 adapter，但 `eval()` 冻结），省一半显存

```yaml
# DPO + LoRA（小模型可以这样玩，大模型几乎必备）
training:
  peft_method: lora
  peft_kwargs:
    rank: 16
    alpha: 32.0
  peft_save_path: checkpoints_dpo/lora_adapter.bin
```

`DPOTask.build_model` 调用 `super().build_model()` 两次（policy + reference），PEFT 路径自动继承——adapter 同时挂载到两个模型，reference 那份通过 `eval()` + `requires_grad=False` 冻结。

---

## 5. 训练监控

### 5.1 日志格式

每个 epoch 总结一行（SFT 例子）：

```text
Epoch  1/1 SUMMARY | Train Loss: 2.3401 | LR: 0.000100 | Time: 4.21s | Peak Mem: 0.00 GB
```

DPO 还会有额外字段：

```text
Epoch  1/1 SUMMARY | Train Loss: 0.6934 | LR: 0.000001 | Time: 12.5s | Peak Mem: 4.21 GB
```

- `Train Loss`：SFT 是 cross-entropy；DPO 是 `-logsigmoid(...)` 平均值，约 0.693 起步（随机），目标是趋近 0
- `LR`：当前学习率
- `Peak Mem`：GPU 峰值显存（CPU 为 0）

### 5.2 TensorBoard

```yaml
logging:
  log_dir: runs/sft_demo
```

```bash
uv run tensorboard --logdir runs/
```

PEFT 训练还会写额外指标：`adalora/effective_rank`（T3 #42）。

### 5.3 验证集

```yaml
data:
  dataset_path: data/sft_train.jsonl
  val_dataset_path: data/sft_val.jsonl   # 可选；不设则跳过验证
```

未设置 `val_dataset_path` → 跳过验证（节省时间）。SFT 和 DPO 都支持独立 val 集。

---

## 6. 故障排除

### 6.1 CUDA OOM

DPO 最容易 OOM（policy + reference）。优先级：

```yaml
training:
  batch_size: 2             # 减小
  gradient_accumulation_steps: 8  # 补有效 batch
optimization:
  gradient_checkpointing: true   # 优先开
  use_amp: true                 # bf16/fp16
data:
  max_seq_len: 512              # 缩短
```

DPO 实在不行：先用 SFT 训一个 ckpt，再在 ckpt 基础上 DPO + LoRA（reference 模型也是 LoRA 化的，省一半显存）。

### 6.2 Loss = NaN / Inf

- 检查 JSONL 是否有空行 / 格式错误（`json.JSONDecodeError` 应在 `SFTDataset._load_data` 抛出）
- DPO learning rate 过大（典型错：`lr=1e-5`；DPO 应是 `5e-7` 量级）
- `dpo_beta` 过小（< 0.01）会导致 log-sigmoid 数值爆炸

### 6.3 `KeyError: 'instruction'`

JSONL 缺字段。SFT 至少要 `instruction` + `output`，`input` 可空字符串。DPO 至少要 `prompt` + `chosen` + `rejected`。

```json
{"instruction": "Q", "input": "", "output": "A"}
{"instruction": "Q", "output": "A"}            # input 可省，模板默认空字符串
```

DPO 字段缺一不可：

```json
{"prompt": "Q", "chosen": "A", "rejected": "B"}
```

### 6.4 Tokenizer 不匹配

预训练时用了 GPT-2 BPE，SFT 一定要 `tokenizer_type: hf` + `tokenizer_path: gpt2`，否则 vocab 会对不上：

```text
RuntimeError: index N is out of bounds for dimension 0 with size V
```

### 6.5 检查点损坏

`CheckpointManager` 写入是**原子操作**（先写 `.tmp` 再 `rename`）。损坏几乎只会因为：

- 磁盘满
- 进程被 kill 在 `.tmp` 阶段（残留 `.tmp` 文件可删）

---

## 7. 从教程到生产

冒烟用的 `sft_local_demo.yaml` 是**冒烟级**。生产通常要：

1. **数据**：Alpaca (52K) / Dolly (15K) / OASST (88K) 的完整子集 + 自定义领域数据
2. **模型**：从上游 SFT ckpt 开始（不要直接拿 base 模型做 DPO）；hidden_size 768-2048，num_layers 12-48
3. **优化**：AMP bf16 + gradient checkpointing + torch.compile（`mode: reduce-overhead`）
4. **并行**：单 GPU 跑 `sft_alpaca.yaml`；多卡 `parallel_strategy: ddp`，多节点 `parallel_strategy: fsdp`
5. **PEFT**：大模型 + 小数据时 `peft_method: lora`，adapter 几 MB，可分发到不同 base 上
6. **DPO 流程**：SFT → DPO + LoRA；不要跳步

详见：

- [架构文档](../reference/architecture.md)
- [PEFT 微调深度指南](../guides/finetuning.md)
- [PEFT serving 集成](../reference/architecture.md#peft-method-registry)
- [推理服务化](./03-inference.md)

---

## 下一步

| 目标 | 文档 |
| --- | --- |
| 推理服务化 | [Tutorials/推理](./03-inference.md) |
| PEFT 深度指南 | [Guides/微调](../guides/finetuning.md) |
| 多卡 / 多节点 | [Guides/分布式训练](../guides/distributed.md) |
| 评估 / lm-eval | [Guides/评估](../guides/evaluation.md) |
| RLHF / PPO | [ROADMAP §阶段十一](../ROADMAP.md) |
