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
uv sync --group streaming
```

---

## 2. 准备数据

创建一个简单的文本文件：

```bash
echo "hello world
this is a test
machine learning is fun
artificial intelligence grows
neural networks process data
deep learning models train" > data/train.txt
```

---

## 3. 训练模型

```bash
# 最简单的训练命令
uv run scripts/train_simple_decoder.py --file-path data/train.txt
```

训练参数自定义：

```bash
uv run scripts/train_simple_decoder.py \
    --file-path data/train.txt \
    --epochs 3 \
    --batch-size 32 \
    --hidden-size 128 \
    --num-layers 4 \
    --save-dir ./checkpoints
```

```bash
# 主流路径：使用 llm-train CLI（推荐生产使用）
uv run llm-train stream_lm --config configs/streaming_local_demo.yaml
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
import torch
from llm.models.decoder import DecoderModel
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer

# 加载训练好的模型
model = DecoderModel(
    vocab_size=1000,  # 替换为你的 vocab 大小
    hidden_size=128,
    num_layers=4,
    num_heads=4,
    max_seq_len=128,
)
model.load_state_dict(torch.load("checkpoints/latest.pt")["model_state_dict"])
model.eval()

# 生成文本
tokenizer = SimpleCharacterTokenizer(corpus=["hello world"])

input_ids = tokenizer.encode("hello").unsqueeze(0)
with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=20)

print(tokenizer.decode(output[0]))
```

---

## 下一步

| 目标             | 文档                                                      |
| ---------------- | --------------------------------------------------------- |
| 完整预训练流程   | [Tutorials/预训练](tutorials/01-pretraining.md)           |
| 微调 (SFT/DPO)  | [Tutorials/微调](tutorials/02-finetuning.md)              |
| 推理服务         | [Tutorials/推理服务](tutorials/03-inference.md)           |
| 评估模型         | [Guides/评估](guides/evaluation.md)                       |
| 量化模型         | [Guides/量化](guides/inference.md#GPTQ-Quantization)      |
| 了解系统架构     | [Architecture](reference/architecture.md)                 |

---

## 常见问题

**Q: GPU 不可用怎么办？**

```bash
uv run scripts/train_simple_decoder.py --file-path data.txt --device cpu
```

**Q: 如何使用多 GPU？**
使用 DDP 模式（详见分布式训练指南）

**Q: 训练中断如何恢复？**

```bash
uv run scripts/train_simple_decoder.py --file-path data.txt --resume ./checkpoints/latest.pt
```
