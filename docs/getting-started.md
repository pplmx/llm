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
git clone https://github.com/your-repo/llm.git
cd llm

# 安装依赖
make init
```

或者使用 uv：

```bash
uv sync
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

---

## 4. 使用模型推理

训练完成后，使用 Python 进行推理：

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

| 目标           | 文档                                            |
| -------------- | ----------------------------------------------- |
| 完整预训练流程 | [Tutorials/预训练](tutorials/01-pretraining.md) |
| 微调现有模型   | [Tutorials/微调](tutorials/02-finetuning.md)    |
| 部署推理服务   | [Tutorials/推理服务](tutorials/03-inference.md) |
| 了解系统架构   | [Architecture](reference/architecture.md)       |

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
