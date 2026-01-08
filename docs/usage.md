# Usage Guide

This guide covers training models and running inference with the `llm` project.

> For installation instructions, see the [README](../README.md#installation).
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

- `--file-path`: Path to the training text file (Required)
- `--val-file-path`: Path to the validation text file
- `--device`: `cpu` or `cuda` (auto-detect by default)
- `--epochs`, `--batch-size`, `--lr`: Training hyperparameters

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

### API Usage

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

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_SERVING_MODEL_PATH` | Path to model checkpoint file | `None` (Dummy Model) |
| `LLM_SERVING_DEVICE` | Computation device (`cpu`, `cuda`, `auto`) | `auto` |
| `LLM_SERVING_API_KEY` | API Key for authentication | `None` (Disabled) |
| `LLM_SERVING_LOG_LEVEL` | Logging level (`INFO`, `DEBUG`, etc.) | `INFO` |

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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | `"llm"` | Model identifier (ignored, for compatibility) |
| `messages` | array | required | List of chat messages |
| `max_tokens` | int | `50` | Maximum tokens to generate |
| `temperature` | float | `1.0` | Sampling temperature (0-2) |
| `top_p` | float | `null` | Nucleus sampling parameter |
| `stream` | bool | `false` | Enable streaming response |
| `presence_penalty` | float | `0.0` | Mapped to repetition_penalty |

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
