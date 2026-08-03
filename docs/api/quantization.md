# `llm.quantization` — Model Quantization

Post-training quantization (PTQ), GPTQ, and AWQ support for reducing
model size and inference latency. The CLI entry point is `llm-quantize`.

## Overview

| Method          | Description                                            |
| --------------- | ------------------------------------------------------ |
| GPTQ            | Greedy row-wise pruning with second-order information  |
| AWQ             | Activation-aware per-channel scales with grid search   |
| Mixed-Precision | Per-layer quantization dispatch via `LayerQuantPolicy` |

## AWQ

::: llm.quantization.awq

## GPTQ

::: llm.quantization.gptq

## Calibration

::: llm.quantization.calibration

## Policies

::: llm.quantization._policy

## PTQ

::: llm.quantization.ptq

## GPTQ Layer

::: llm.quantization._gptq_layer

## AWQ Layer

::: llm.quantization._awq_layer
