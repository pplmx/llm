---
tags:
  - 快速开始
  - 入门
---

# 快速开始

5 分钟内启动并运行你的第一个 LLM 训练。

## 目标

本指南让你能够：

- 安装环境
- 训练第一个模型
- 生成文本

---

## 1. 安装

```bash
# 克隆项目
git clone https://github.com/pplmx/llm.git
cd llm

# 安装依赖
make init
```

或者使用 uv（默认包含 test 依赖组, 可直接 `make test`）:

```bash
uv sync
```

流式预训练等可选能力:

```bash
uv sync --extra streaming
```

---

## 2. 准备数据

创建一个简单的文本文件（**每行一条样本**，流式 demo 配置默认读 `data/demo.txt`）：

```bash
echo "hello world
this is a test
machine learning is fun
artificial intelligence grows
neural networks process data
deep learning models train" > data/demo.txt
```

---

## 3. 训练模型

主流路径：`llm-train` CLI（推荐生产使用），跑仓库自带的流式预训练冒烟配置：

```bash
# 流式预训练（CPU 几秒完成，走 streaming → forward → backward → checkpoint 全链路）
uv run llm-train --task stream_lm --config-path configs/streaming_local_demo.yaml
```

命令行覆盖 YAML 字段（实验 sweep 常用）：

```bash
uv run llm-train --task stream_lm --config-path configs/streaming_local_demo.yaml \
    --epochs 3 \
    --steps-per-epoch 20 \
    --batch-size 8 \
    --lr 5e-4
```

替代方案：单文件 demo 脚本（仅作最小演示，不用于生产）：

```bash
uv run scripts/train_simple_decoder.py --file-path data/demo.txt
```

---

## 4. 使用模型推理

训练完成后，可通过以下方式使用模型：

### 方式一：启动推理服务（推荐）

```bash
# 启动推理服务（推荐）
uv run llm-serve

# 然后通过 OpenAI 兼容 API 调用
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llm","messages":[{"role":"user","content":"hello"}],"max_tokens":10}'
```

### 方式二：Python 代码调用

```python
from llm.models.decoder import DecoderModel
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer
from llm.training.core.checkpoint import load_checkpoint_payload

# 加载训练好的模型
model = DecoderModel(
    vocab_size=1000,  # 替换为你的 vocab 大小
    hidden_size=128,
    num_layers=4,
    num_heads=4,
    max_seq_len=128,
)
# v2 布局：checkpoints/latest.safetensors + .meta.json + .extra_state.pt
ckpt = load_checkpoint_payload("checkpoints/latest")
model.load_state_dict(ckpt["model_state"])
model.eval()

# 生成文本（generation 走 llm.generation 的 generate 入口，DecoderModel 本身没有 .generate）
from llm.generation import generate

tokenizer = SimpleCharacterTokenizer(corpus=["hello world"])
output = generate(model, tokenizer, "hello", max_new_tokens=20)

print(output)
```

---

## 下一步

| 目标           | 文档                                            |
| -------------- | ----------------------------------------------- |
| 完整预训练流程 | [Tutorials/预训练](tutorials/01-pretraining.md) |
| 微调 (SFT/DPO) | [Tutorials/微调](tutorials/02-finetuning.md)    |
| 推理服务       | [Tutorials/推理服务](tutorials/03-inference.md) |
| 评估模型       | [Guides/评估](guides/evaluation.md)             |
| 量化模型       | [Guides/量化](guides/quantization.md)           |
| 了解系统架构   | [Architecture](reference/architecture.md)       |

---

## 常见问题

**Q: GPU 不可用怎么办？**

```bash
uv run llm-train --task stream_lm --config-path configs/streaming_local_demo.yaml
# （demo 配置默认 CPU 可跑；AMP / NCCL 需要 GPU）
```

**Q: 如何使用多 GPU？**
使用 DDP 模式（详见分布式训练指南）

**Q: 训练中断如何恢复？**

在 YAML 的 `checkpoint` 段设置 resume 路径（`llm-train` 没有 resume 的 CLI 参数）：

```bash
uv run llm-train --task stream_lm --config-path configs/streaming_local_demo.yaml
# checkpoint:
#   resume_from_checkpoint: checkpoints/latest
```
