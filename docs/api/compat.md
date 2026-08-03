# `llm.compat` — Compatibility Layer

HuggingFace compatibility utilities for loading and publishing models.
The `compat` group (`uv sync --extra compat`) provides optional
dependencies (`huggingface_hub`, `pillow`, `safetensors`).

## Overview

| Module           | Purpose                           |
| ---------------- | --------------------------------- |
| `hf_loader`      | Load HuggingFace checkpoints      |
| `hf_publisher`   | Publish models to HuggingFace Hub |
| `weight_mapping` | Map weight names between formats  |

## HF Loader

::: llm.compat.hf_loader

## HF Publisher

::: llm.compat.hf_publisher

## Weight Mapping

::: llm.compat.weight_mapping
