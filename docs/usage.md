# Usage Guide

This guide covers training models and running inference with the `llm` project.

> For installation instructions, see the main [README](https://github.com/pplmx/llm/blob/main/README.md#installation).
> For development setup (testing, linting, Docker), see the [Development Guide](development.md).

## Training

### Using the CLI

The recommended way to train models is using the `llm-train` CLI:

```bash
llm-train --task <task_name> [options]
```

Use `--help` to see all available options:

```bash
llm-train --help
```

**Examples:**

```bash
# Regression task (uses synthetic data, works out of the box)
llm-train --task regression --epochs 10 --batch-size 32

# Language modeling task (requires dataset configuration)
# Note: The lm task uses TextDataModule which needs a configured dataset.
# Use a config file or the standalone script below for LM training.
llm-train --task lm --config-path configs/example.yaml --epochs 5
```

> [!NOTE]
> The `--task lm` option requires dataset configuration via a YAML config file.
> For quick experimentation, use the standalone decoder training script instead.

### Standalone Simple Decoder Training

For a simple example of training a decoder-only model on a text file:

```bash
uv run scripts/train_simple_decoder.py --file-path data/dummy_corpus.txt --epochs 5
```

**Common Options:**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--file-path` | Path | **必需** | 训练文本文件路径 |
| `--val-file-path` | Path | None | 验证文本文件路径 |
| `--device` | str | auto | `cpu` 或 `cuda` |
| `--epochs` | int | 1 | 训练轮数 |
| `--batch-size` | int | 16 | 批次大小 |
| `--lr` | float | 1e-3 | 学习率 |
| `--hidden-size` | int | 64 | 模型隐藏层大小 |
| `--num-layers` | int | 2 | Transformer 层数 |
| `--num-heads` | int | 2 | 注意力头数 |
| `--max-seq-len` | int | 32 | 最大序列长度 |

#### Checkpoint 功能

支持训练过程中自动保存和恢复 checkpoint：

```bash
# 训练并自动保存 checkpoint (每100步保存一次)
uv run scripts/train_simple_decoder.py \
    --file-path data/train.txt \
    --save-dir ./checkpoints \
    --save-interval 100 \
    --epochs 3

# 从 checkpoint 恢复训练
uv run scripts/train_simple_decoder.py \
    --file-path data/train.txt \
    --resume ./checkpoints/latest.pt \
    --epochs 3
```

**Checkpoint 参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--save-dir` | Path | `./checkpoints` | checkpoint 保存目录 |
| `--save-interval` | int | 100 | 每 N 步保存一次 |
| `--resume` | Path | None | 从 checkpoint 恢复 |

**Checkpoint 包含：**
- 模型权重 (`model_state_dict`)
- 优化器状态 (`optimizer_state_dict`)
- 训练轮数、步数、loss
- 模型配置 (用于恢复训练)

**优雅退出：**
支持 Ctrl+C 中断训练，系统会自动保存当前 checkpoint 后再退出。

## Inference

To generate text using a trained model, you can use the `generate` function from `src/llm/inference.py`.

### Python API Example (Simple)

Here's a basic example using the built-in `SimpleCharacterTokenizer`:

```python
import torch
from llm.models.decoder import DecoderModel
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer
from llm.inference import generate

# 1. Create a simple tokenizer
corpus = ["Hello world", "This is a test"]
tokenizer = SimpleCharacterTokenizer(corpus)

# 2. Initialize Model
# Configuration should match the tokenizer's vocab size
model = DecoderModel(
    vocab_size=tokenizer.vocab_size,
    hidden_size=64,
    num_layers=2,
    num_heads=4,
    max_seq_len=128,
)
# Load weights here if available
# model.load_state_dict(torch.load("path/to/model.pt"))

# 3. Generate Text
generated_text = generate(
    model=model,
    tokenizer=tokenizer,
    prompt="Hello",
    max_new_tokens=20,
    temperature=0.8
)
print(generated_text)
```

### Python API Example (HuggingFace)

For production use with pre-trained tokenizers, use `HFTokenizer`:

```python
import torch
from llm.models.decoder import DecoderModel
from llm.tokenization.tokenizer import HFTokenizer
from llm.inference import generate

# 1. Load Tokenizer (e.g., GPT-2 from HuggingFace)
# Note: This requires transformers library: pip install transformers
tokenizer = HFTokenizer.from_pretrained("gpt2")

# 2. Initialize Model
# IMPORTANT: vocab_size must match the tokenizer
model = DecoderModel(
    vocab_size=tokenizer.vocab_size,  # GPT-2: 50257
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    max_seq_len=1024,
)

# 3. Load trained weights (if available)
# checkpoint = torch.load("path/to/checkpoint.pt")
# model.load_state_dict(checkpoint["model_state_dict"])

# 4. Generate Text
generated_text = generate(
    model=model,
    tokenizer=tokenizer,
    prompt="Once upon a time",
    max_new_tokens=50,
    temperature=0.9,
    top_p=0.95
)
print(generated_text)
```

### Streaming Generation

For real-time streaming output, use `stream_generate`:

```python
from llm.inference import stream_generate

# Generate with streaming (yields tokens as they are generated)
for token in stream_generate(
    model=model,
    tokenizer=tokenizer,
    prompt="Hello",
    max_new_tokens=20,
    temperature=0.8
):
    print(token, end="", flush=True)
```

### Batch Generation

For processing multiple prompts efficiently:

```python
from llm.inference import batch_generate

# Generate multiple prompts in a single forward pass
results = batch_generate(
    model=model,
    tokenizer=tokenizer,
    prompts=["Hello world", "Once upon a time", "The quick brown fox"],
    max_new_tokens=20,
    temperature=0.8
)
# Returns: List of generated strings
```

## Inference Serving

This project includes a production-ready REST API for inference service, built with FastAPI.

### Features

- **Streaming Support**: Server-Sent Events (SSE) for real-time token generation.
- **Advanced Sampling**: Support for `top_p` (Nucleus Sampling) and `repetition_penalty`.
- **Production Ready**: Structured logging, Prometheus metrics, and API Key authentication.

### Starting the Server

**Using CLI (Recommended):**

```bash
llm-serve
```

**Using Docker:**

```bash
make image
make compose-up
```

### Server Configuration

The inference server can be configured via environment variables:

| Environment Variable                  | Default | Description                      |
| ------------------------------------- | ------- | -------------------------------- |
| `LLM_SERVING_MODEL_PATH`              | -       | Path to model checkpoint         |
| `LLM_SERVING_TOKENIZER_PATH`          | -       | Path to tokenizer                |
| `LLM_SERVING_DEVICE`                  | `auto`  | Device: `cpu`, `cuda`, or `auto` |
| `LLM_SERVING_API_KEY`                 | -       | API key for authentication       |
| `LLM_SERVING_LOG_LEVEL`               | `INFO`  | Logging level                    |
| `LLM_SERVING_COMPILE_MODEL`           | `false` | Enable `torch.compile`           |
| `LLM_SERVING_MAX_CONCURRENT_REQUESTS` | `4`     | Max concurrent requests          |
| `LLM_SERVING_REQUEST_TIMEOUT`         | `60.0`  | Request timeout (seconds)        |

**Model Architecture Params (for dummy init):**

| Environment Variable       | Default | Description                  |
| -------------------------- | ------- | ---------------------------- |
| `LLM_SERVING_HIDDEN_SIZE`  | `64`    | Model hidden size            |
| `LLM_SERVING_NUM_LAYERS`   | `2`     | Number of transformer layers |
| `LLM_SERVING_NUM_HEADS`    | `4`     | Number of attention heads    |
| `LLM_SERVING_MAX_SEQ_LEN`  | `128`   | Maximum sequence length      |
| `LLM_SERVING_NUM_KV_HEADS` | -       | KV heads (for GQA)           |
| `LLM_SERVING_USE_MOE`      | `false` | Enable MoE                   |
| `LLM_SERVING_NUM_EXPERTS`  | `0`     | Number of MoE experts        |
| `LLM_SERVING_TOP_K`        | `0`     | Top-k experts for MoE        |
| `LLM_SERVING_ATTN_IMPL`    | `mha`   | Attention implementation     |
| `LLM_SERVING_MLP_IMPL`     | `mlp`   | MLP implementation           |

**Example:**

```bash
LLM_SERVING_MODEL_PATH=./checkpoint.pt \
LLM_SERVING_COMPILE_MODEL=true \
LLM_SERVING_MAX_CONCURRENT_REQUESTS=8 \
llm-serve
```

### API Usage

#### GET /health

Check if the inference server is running and healthy.

**Response:**

```json
{
  "status": "ok"
}
```

#### POST /generate

Generate text from a prompt.

**Request Body:**

```json
{
  "prompt": "Hello, world",
  "max_new_tokens": 50,
  "temperature": 0.8,
  "top_k": 5,
  "top_p": 0.9,
  "repetition_penalty": 1.1,
  "stream": false
}
```

**Streaming Request:**

Set `"stream": true` to receive a stream of tokens.

```bash
curl -X POST "http://127.0.0.1:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Tell me a story", "stream": true}'
```

**Response (Non-streaming):**

```json
{
  "generated_text": "Hello, world! This is a generated text...",
  "token_count": 12
}
```

#### POST /batch_generate

Generate text for multiple prompts in a single request.

**Request Body:**

```json
{
  "prompts": ["Hello", "Once upon a time", "The quick brown fox"],
  "max_new_tokens": 50,
  "temperature": 0.8,
  "top_k": 5,
  "top_p": 0.9,
  "repetition_penalty": 1.1
}
```

**Response:**

```json
{
  "results": [
    {"generated_text": "Hello! How can I help?", "token_count": 5},
    {"generated_text": "Once upon a time in a distant land...", "token_count": 12},
    {"generated_text": "The quick brown fox jumps over the lazy dog.", "token_count": 10}
  ]
}
```

### Authentication

If `LLM_SERVING_API_KEY` is set, you must provide the key in the `X-API-Key` header.

```bash
export LLM_SERVING_API_KEY="my-secret-key"
# Start server...

curl -X POST "http://127.0.0.1:8000/generate" \
     -H "X-API-Key: my-secret-key" \
     ...
```

### Configuration

You can configure the serving engine using environment variables:

| Variable                 | Description                                | Default              |
| ------------------------ | ------------------------------------------ | -------------------- |
| `LLM_SERVING_MODEL_PATH` | Path to model checkpoint file              | `None` (Dummy Model) |
| `LLM_SERVING_DEVICE`     | Computation device (`cpu`, `cuda`, `auto`) | `auto`               |
| `LLM_SERVING_API_KEY`    | API Key for authentication                 | `None` (Disabled)    |
| `LLM_SERVING_LOG_LEVEL`  | Logging level (`INFO`, `DEBUG`, etc.)      | `INFO`               |

### Metrics

Prometheus metrics are available at `/metrics`.

```bash
curl http://127.0.0.1:8000/metrics
```

### Performance Benchmarking

A benchmark script is provided to measure inference performance (Latency and TPS).

```bash
# Run benchmark with torch.compile enabled
uv run scripts/benchmark_inference.py --compile --runs 5
```

Arguments:

- `--runs`: Number of benchmark iterations.
- `--compile`: Enable `torch.compile` optimization.
- `--device`: Target device (e.g., `cuda`).
- `--max_new_tokens`: Number of tokens to generate per run.

## OpenAI-Compatible API

The serving module provides an OpenAI-compatible endpoint, allowing you to use the official `openai` Python SDK.

### Endpoint

`POST /v1/chat/completions`

### Request Body

```json
{
  "model": "llm",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 50,
  "temperature": 0.8,
  "stream": false
}
```

**Parameters:**

| Parameter           | Type         | Default  | Description                                   |
| ------------------- | ------------ | -------- | --------------------------------------------- |
| `model`             | string       | `"llm"`  | Model identifier (ignored, for compatibility) |
| `messages`          | array        | required | List of chat messages                         |
| `max_tokens`        | int          | `50`     | Maximum tokens to generate                    |
| `temperature`       | float        | `1.0`    | Sampling temperature (0-2)                    |
| `top_p`             | float        | `null`   | Nucleus sampling parameter                    |
| `top_k`             | int          | `null`   | Top-k sampling parameter                      |
| `stream`            | bool         | `false`  | Enable streaming response                     |
| `presence_penalty`  | float        | `0.0`    | Mapped to repetition_penalty                  |
| `frequency_penalty` | float        | `0.0`    | Frequency penalty (not implemented, reserved) |
| `stop`              | string/array | `null`   | Stop sequences                                |

### Response

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1704556800,
  "model": "llm",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Hello! How can I help?"},
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 10,
    "total_tokens": 30
  }
}
```

### Authentication

Supports both `X-API-Key` header and Bearer token:

```bash
# Using X-API-Key
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "X-API-Key: your-key" \
     -H "Content-Type: application/json" \
     -d '{"messages": [{"role": "user", "content": "hi"}]}'

# Using Bearer token (OpenAI SDK style)
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Authorization: Bearer your-key" \
     -H "Content-Type: application/json" \
     -d '{"messages": [{"role": "user", "content": "hi"}]}'
```

### Using OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-key"  # or any string if auth is disabled
)

response = client.chat.completions.create(
    model="llm",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    max_tokens=50,
    temperature=0.8
)

print(response.choices[0].message.content)
```

### Streaming with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="test")

stream = client.chat.completions.create(
    model="llm",
    messages=[{"role": "user", "content": "Tell me a story"}],
    max_tokens=100,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```
