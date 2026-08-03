# `llm.core.peft` — Parameter-Efficient Fine-Tuning

The PEFT subpackage implements parameter-efficient fine-tuning methods
that train only a small set of additional parameters while keeping the
pretrained model frozen. All methods register into `PEFT_REGISTRY` and
can be applied via the training task configuration.

## Overview

| Method           | Paper                | Trainable Parameters   |
| ---------------- | -------------------- | ---------------------- |
| LoRA             | Hu et al. 2021       | Low-rank decomposition |
| QLoRA            | Dettmers et al. 2023 | 4-bit NF4 + LoRA       |
| AdaLoRA          | He et al. 2022       | Adaptive rank LoRA     |
| Prefix Tuning    | Li & Liang 2021      | Virtual prefix tokens  |
| IA³              | Liu et al. 2021      | Multiplicative scaling |
| BitFit           | Zaken et al. 2021    | Bias-only              |
| Adapter          | Houlsby et al. 2019  | Bottleneck layers      |
| Pfeiffer Adapter | Pfeiffer et al. 2021 | FFN-only bottleneck    |

## Registry

::: llm.core.peft.registry

## Method Types

::: llm.core.peft.types

## Built-in Methods

::: llm.core.peft.methods

## Checkpoint Helpers

::: llm.core.peft.checkpoint
