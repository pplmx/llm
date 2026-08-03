# `llm.quantization` — Model Quantization

Post-training quantization (PTQ) and GPTQ support for reducing model
size and inference latency. The CLI entry point is `llm-quantize`.

## Overview

| Method          | Description                                            |
| --------------- | ------------------------------------------------------ |
| GPTQ            | Greedy row-wise pruning with second-order information  |
| Mixed-Precision | Per-layer quantization dispatch via `LayerQuantPolicy` |

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
