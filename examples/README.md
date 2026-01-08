# Examples

This directory contains minimal examples for common use cases.

## Quick Start

### Basic Inference

```bash
python examples/inference_demo.py
```

### Using OpenAI SDK

```bash
# Start server first
llm-serve

# In another terminal
python examples/openai_client_demo.py
```

### KV Cache for Efficient Generation

```bash
python examples/kv_cache_demo.py
```

### QLoRA Fine-Tuning

```bash
python examples/qlora_finetuning_demo.py
```

## Files

| File | Description |
| ---- | ----------- |
| `inference_demo.py` | Basic text generation with DecoderModel |
| `openai_client_demo.py` | Using OpenAI SDK with llm-serve |
| `kv_cache_demo.py` | Efficient generation with pre-allocated KVCache |
| `qlora_finetuning_demo.py` | Memory-efficient fine-tuning with 4-bit quantization |
