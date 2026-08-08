# `llm.core` — Building Blocks

Reusable, framework-agnostic building blocks shared by training and
serving. Everything here is pure PyTorch (no FastAPI, no trainer
plumbing) so it can be reused in notebooks and other runners.

## KV Cache

::: llm.core.kv_cache

## Attention Implementations

::: llm.core.attn.mha
::: llm.core.attn.sdpa
::: llm.core.attn.mla
::: llm.core.attn.flash_attn
::: llm.core.attn.base

## Paged Attention

Block-allocator KV cache for serving (see
[ADR-004](../adr/004-paged-attention-serving.md)):

::: llm.core.paged_attention.paged_kv_cache
::: llm.core.paged_attention.attention
::: llm.core.paged_attention.block_allocator
::: llm.core.paged_attention.block_manager

## MLP Variants

::: llm.core.mlp
::: llm.core.moe.moe

## Transformer Block

::: llm.core.transformer_block

## Normalization Layers

::: llm.core.rms_norm
::: llm.core.layer_norm

## Embeddings and Positional Encoding

::: llm.core.embedding
::: llm.core.positional_encoding
::: llm.core.rope
::: llm.core.alibi

## PEFT Helpers

The unified `PEFT_REGISTRY` lives in
[`llm.core.peft`](peft.md); these are the standalone helper modules for
the individual methods.

::: llm.core.lora
::: llm.core.qlora
::: llm.core.adalora
::: llm.core.bitfit
::: llm.core.ia3
::: llm.core.adapter
::: llm.core.pfeiffer_adapter
::: llm.core.prefix_tuning

## Component Registry

::: llm.core.registry
