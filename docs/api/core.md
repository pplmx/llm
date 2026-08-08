# `llm.core` — Building Blocks

Reusable, framework-agnostic building blocks shared by training and
serving. Everything here is pure PyTorch (no FastAPI, no trainer
plumbing) so it can be reused in notebooks and other runners.

## KV Cache

::: llm.core.kv_cache

## Attention Implementations

::: llm.core.attn.mha

## MLP Variants

::: llm.core.mlp

## Normalization Layers

::: llm.core.rms_norm
::: llm.core.layer_norm

## Embeddings and Positional Encoding

::: llm.core.embedding
::: llm.core.positional_encoding
::: llm.core.rope
::: llm.core.alibi

## LoRA / QLoRA

::: llm.core.lora

## Component Registry

::: llm.core.registry
