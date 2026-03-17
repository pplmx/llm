# Examples

本目录包含常见使用场景的最小示例，演示 llm 项目的核心功能。

## 快速开始

### 1. 基础推理

```bash
python examples/inference_demo.py
```

功能：使用 DecoderModel + SimpleCharacterTokenizer 进行基础文本生成。

### 2. OpenAI SDK 调用

```bash
# 终端1: 启动推理服务
llm-serve

# 终端2: 运行客户端
python examples/openai_client_demo.py
```

功能：通过 OpenAI SDK 调用 llm-serve 推理服务，支持流式输出。

### 3. KV Cache 高效推理

```bash
python examples/kv_cache_demo.py
```

功能：使用预分配 KV Cache 进行高效自回归生成，支持 GQA (分组查询注意力)。

### 4. QLoRA 高效微调

```bash
python examples/qlora_finetuning_demo.py
```

功能：4-bit NF4 量化 + LoRA 适配器，显存减少约 4 倍。

## 示例文件详解

| 文件                       | 功能                   | 关键依赖                                         |
| -------------------------- | ---------------------- | ------------------------------------------------ |
| `inference_demo.py`        | 基础文本生成           | DecoderModel, SimpleCharacterTokenizer, generate |
| `openai_client_demo.py`    | OpenAI 兼容 API 客户端 | openai SDK, SSE 流式                             |
| `kv_cache_demo.py`         | KV Cache 高效推理      | KVCache, GQA (num_kv_heads)                      |
| `qlora_finetuning_demo.py` | QLoRA 微调             | apply_qlora, NF4 量化, LoRA 适配器               |

## 运行示例

### 环境准备

```bash
# 安装项目依赖
make init

# 激活虚拟环境
source .venv/bin/activate
```

### 推理示例

```bash
# 基础推理
python examples/inference_demo.py

# KV Cache 推理
python examples/kv_cache_demo.py

# 带服务的推理
llm-serve &
python examples/openai_client_demo.py
```

### 微调示例

```bash
# QLoRA 微调
python examples/qlora_finetuning_demo.py
```

## 进阶使用

如需更复杂的使用场景，请参考：

- 训练文档: `docs/training/README.md`
- 微调指南: `docs/guide-finetuning.md`
- 推理指南: `docs/guide-inference.md`
