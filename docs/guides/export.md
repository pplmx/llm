---
tags:
  - 指南
  - 导出
---

# 模型导出指南

本文介绍如何把训练好的模型导出为 ONNX、TorchScript 或 GGUF，以及如何注册
自定义导出后端。

> 对应 API 文档见 [llm.export](../api/export.md)，架构决策见
> [ADR-005 Export Registry](../adr/005-export-registry.md) 与
> [ADR-011 GGUF](../adr/011-gguf-export.md)。

## 格式总览

| 格式        | 典型用途                            | 量化选项                | 状态                   |
| ----------- | ----------------------------------- | ----------------------- | ---------------------- |
| ONNX        | 跨运行时推理（onnxruntime 等）      | -                       | 参考实现，API 稳定     |
| TorchScript | PyTorch C++ / 服务端部署            | -                       | trace 路径可用         |
| GGUF        | llama.cpp 等 GGML 系运行时          | F16 / F32 / Q4_0 / Q8_0 | v1（ADR-011）          |
| 自定义      | 通过 `llm.export_backends` 插件注册 | 由后端决定              | `EXPORT_REGISTRY` 机制 |

## 统一入口：`export_model`

所有内置格式通过 [EXPORT_REGISTRY](../api/export.md) 统一调度，入口是
`llm.export.export_model(name, model, output_path, **kwargs)`：

```python
from llm.export import export_model

# ONNX
export_model("onnx", model, "model.onnx")

# TorchScript（默认 trace 模式）
export_model("torchscript", model, "model.pt", method="trace")

# GGUF（默认 F16；可选 q4_0 / q8_0 块量化）
export_model("gguf", model, "model.gguf", quantize="q4_0")
```

## GGUF：为 llama.cpp 生态导出

GGUF 导出（`llm.export.gguf`）不依赖 torch 之外的重型依赖，核心格式层是
torch-free 的。支持：

- 张量类型：F32 / F16 原样导出，Q4_0 / Q8_0 按 32 元素块量化
  （与 ggml 参考实现字节兼容）；
- 标准 `general.*` 元数据，可用 `metadata=` 覆盖；
- 非浮点张量会被显式拒绝；导出采用原子写入（临时文件 + rename）。

```python
from llm.export import export_to_gguf

path = export_to_gguf(model, "model-q4.gguf", quantize="q4_0", model_name="my-model")
```

导出的 GGUF 文件可直接交给 llama.cpp 等 GGML 系运行时加载。

> v1 限制：K-quants / Q4_1 / IQ 类型、mmap 读取、llama.cpp 架构张量命名
> （`blk.*`）与 tokenizer 元数据是后续规划，暂未覆盖。

## TorchScript：trace 优先

`export_model("torchscript", ...)` 默认使用 `method="trace"`，把模型按示例输入
固化导出。`method="script"` 对 `DecoderModel` 的完整 scripting 支持仍在推进中
（`PositionalEncoding` 等模块存在已知限制），生产路径请使用 trace 模式。

```python
export_model("torchscript", model, "model.pt", method="trace", input_shape=(1, 64))
```

## ONNX：参考实现

`llm.export.onnx` 是导出层的参考实现，附带验证工具：

```python
from llm.export import export_to_onnx, verify_onnx, get_onnx_info

export_to_onnx(model, "model.onnx", input_shape=(1, 64))
verify_onnx("model.onnx")  # 加载并检查图
info = get_onnx_info("model.onnx")  # 输入/输出签名等
```

## 注册自定义导出后端

第三方导出目标（TensorRT-LLM、vLLM 等）通过 `llm.export_backends` entry-point
组注册，工厂签名与内置后端一致（`build_<target>_exporter(model, output_path, **kwargs)`）：

```toml
[project.entry-points."llm.export_backends"]
my-backend = "my_pkg.exporter:build_my_exporter"
```

```python
from llm.export import export_model

export_model("my-backend", model, "model.bin", option="value")
```

## 与量化、发布的衔接

- 量化后的模型（`GPTQQuantizedLinear` 等）是 `DecoderModel` 子类，可直接导出；
- 导出前请先用 [量化指南](quantization.md) 或直接 fp16 权重，按目标运行时
  的支持范围选择格式；
- 如需发布到 HuggingFace Hub，用 `llm.compat.hf_publisher.push_to_hub`，
  见 [compat API](../api/compat.md)。

## 进一步阅读

- API：`llm.export`（[export.md](../api/export.md)）
- 架构决策：[ADR-005 Export Registry](../adr/005-export-registry.md)、
  [ADR-011 GGUF](../adr/011-gguf-export.md)
- FAQ：[导出相关问题](../faq.md#导出-export)
