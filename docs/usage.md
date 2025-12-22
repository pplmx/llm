# Usage Guide

This guide provides instructions on how to install, develop, train, and run inference with the `llm` project.

## Installation

To set up the project environment, ensure you have `uv` installed. Then run:

```bash
make init
```

This command initializes the virtual environment and installs pre-commit hooks. Alternatively, you can run:

```bash
uv sync
```

## Development Commands

The project uses `Makefile` to simplify common development tasks.

- **Run Tests**:

  ```bash
  make test
  ```

- **Lint Code**:

  ```bash
  make lint
  ```

- **Format Code**:

  ```bash
  make fmt
  ```

- **Type Check**:

  ```bash
  make ty
  ```

- **View Allure Report**:

  ```bash
  make allure
  ```

- **Clean Build Artifacts**:

  ```bash
  make clean
  ```

## Training

### Using the Training Script

The main training entry point is `src/llm/training/train.py`. You can run it using `uv run`.

Syntax:

```bash
uv run -m llm.training.train --task <task_name> [options]
```

Example (Regression Task):

```bash
uv run -m llm.training.train --task regression --training-epochs 10
```

### Standalone Simple Decoder Training

For a simple example of training a decoder-only model on a text file, use `scripts/train_simple_decoder.py`.

```bash
uv run scripts/train_simple_decoder.py --file_path <path_to_text_file> [options]
```

**Common Options:**

- `--file_path`: Path to the training text file (Required).
- `--val_file_path`: Path to the validation text file.
- `--device`: `cpu` or `cuda`.
- `--epochs`: Number of training epochs.
- `--batch_size`: Batch size.

Example:

```bash
uv run scripts/train_simple_decoder.py --file_path data/corpus.txt --epochs 5 --device cuda
```

## Inference

To generate text using a trained model, you can use the `generate` function from `src/llm/inference.py`.

### Python API Example

```python
import torch
from llm.models.decoder import DecoderModel
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer
from llm.inference import generate

# 1. Load Tokenizer
# Ensure you use the same corpus/vocab as training
corpus = ["your training data strings"]
tokenizer = SimpleCharacterTokenizer(corpus)

# 2. Initialize Model
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
