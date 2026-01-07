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

## Files

| File | Description |
| ------ | ------------- |
| `inference_demo.py` | Basic text generation with DecoderModel |
| `openai_client_demo.py` | Using OpenAI SDK with llm-serve |
