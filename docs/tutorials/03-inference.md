# 推理教程

训练完成后如何使用模型生成文本。

## 概述

本教程涵盖：

- 基础推理
- 采样策略
- 推理服务部署

---

## 1. 基础推理

### Python API

```python
import torch
from llm.models.decoder import DecoderModel
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer
from llm.inference import generate

# 1. 加载模型和分词器
model = DecoderModel(
    vocab_size=1000,
    hidden_size=256,
    num_layers=4,
    num_heads=4,
    max_seq_len=128,
)
model.load_state_dict(torch.load("checkpoint.pt")["model_state_dict"])
model.eval()

tokenizer = SimpleCharacterTokenizer(corpus=["hello world"])

# 2. 生成文本
input_ids = tokenizer.encode("hello")
output = generate(
    model,
    input_ids,
    max_new_tokens=50,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
)

print(tokenizer.decode(output))
```

---

## 2. 生成参数

| 参数                 | 类型  | 默认值 | 说明                 |
| -------------------- | ----- | ------ | -------------------- |
| `max_new_tokens`     | int   | 50     | 生成的最大 token 数  |
| `temperature`        | float | 1.0    | 采样温度，越高越随机 |
| `top_k`              | int   | None   | top-k 采样           |
| `top_p`              | float | None   | nucleus 采样         |
| `repetition_penalty` | float | 1.0    | 重复惩罚             |

### 采样策略选择

| 场景     | 建议参数                        |
| -------- | ------------------------------- |
| 创意写作 | temperature=0.8-1.0             |
| 精确问答 | temperature=0.1-0.3, top_p=0.95 |
| 代码生成 | temperature=0.2, top_p=0.95     |

---

## 3. 推理服务

### 启动服务

```bash
uv run python -m llm.serving.api
```

服务启动后访问 `http://localhost:8000/docs` 查看 API。

### API 端点

```bash
# 单句生成
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_new_tokens": 50,
    "temperature": 0.7
  }'

# 流式生成
curl -N "http://localhost:8000/generate/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello world",
    "stream": true
  }'
```

---

## 4. 推理优化

### KV Cache

自动启用以加速自回归生成：

```python
# 使用 KV Cache (默认启用)
output = generate(model, input_ids, use_cache=True)
```

### 批量推理

```python
# 批量生成
prompts = ["hello", "world", "test"]
input_ids_list = [tokenizer.encode(p) for p in prompts]

# 批量处理
outputs = []
for ids in input_ids_list:
    output = generate(model, ids)
    outputs.append(output)
```

### 详细优化指南

请参考 [Inference Optimization](../guides/inference.md)

---

## 下一步

| 目标     | 文档                                      |
| -------- | ----------------------------------------- |
| 性能优化 | [Guides/推理优化](../guides/inference.md) |
