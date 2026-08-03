# `llm.export` — Model Export

Export models to ONNX and TorchScript formats. Supports custom export
backends registered via the `llm.export_backends` entry point group.

## Overview

| Format      | Status                               |
| ----------- | ------------------------------------ |
| ONNX        | Reference implementation, stable API |
| TorchScript | Entry-point registered               |
| Custom      | Via `EXPORT_REGISTRY` plugin points  |

## Export Registry

::: llm.export.registry

## ONNX Export

::: llm.export.onnx

## TorchScript Export

::: llm.export.torchscript

## Shared Wrapper

::: llm.export._wrapper
