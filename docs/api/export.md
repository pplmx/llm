# `llm.export` — Model Export

Export models to ONNX, TorchScript, and GGUF formats. Supports custom
export backends registered via the `llm.export_backends` entry point
group.

## Overview

| Format      | Status                                                     |
| ----------- | ---------------------------------------------------------- |
| ONNX        | Reference implementation, stable API                       |
| TorchScript | Entry-point registered                                     |
| GGUF        | Entry-point registered (v1: F16/F32/Q4_0/Q8_0, ADR-011)    |
| Custom      | Via `EXPORT_REGISTRY` plugin points                        |

## Export Registry

::: llm.export.registry

## ONNX Export

::: llm.export.onnx

## TorchScript Export

::: llm.export.torchscript

## GGUF Export

::: llm.export.gguf

## Shared Wrapper

::: llm.export._wrapper
