# 016. TensorRT and Core ML: what the export path can and cannot support

Date: 2026-11-18

## Status

Proposed

## Context

ROADMAP §14.2「模型导出」有两项未做：

```text
- [ ] 支持 TensorRT 优化
- [ ] 探索 Core ML 支持 (iOS 部署)
```

两项都是一句话，没有任何设计。本 ADR 记录**可行性调查的实测结果**，把「能做什么、
代价是什么、什么做不了」写清楚，再决定要不要做。

### 实测环境（事实）

| 项                                            | 值                                                  |
| --------------------------------------------- | --------------------------------------------------- |
| GPU                                           | 8 × NVIDIA A100-SXM4-80GB                           |
| 驱动                                          | 595.45.04（报告支持 CUDA 13.2）                     |
| torch                                         | 2.14.1+**cu130**，arch list 含 `sm_80`              |
| Python                                        | 3.14.5                                              |
| `onnxruntime`                                 | 1.31.0 —— providers 仅 **CPU / Azure**，**无 CUDA** |
| `tensorrt` / `torch_tensorrt` / `coremltools` | 均未安装                                            |

`uv pip install --dry-run tensorrt` 能解析出 `tensorrt==11.5.0.63` +
`tensorrt-cu13{,-bindings,-libs}`，即存在匹配 cu130 与 Python 3.14 的 wheel。
实测安装未在合理时间内完成（数 GB 的 CUDA 库下载），因此**本轮没有拿到 TensorRT
真正运行的结果**——下文凡涉及 TensorRT 运行时行为的判断都标 `[INFERENCE]`。

### ONNX 产物质量（实测）

用 `export_to_onnx` 导出 `DecoderModel(vocab_size=64, hidden=32, layers=2, heads=2)`：

- opset **17**，IR version 8，文件 158.7 KiB
- 23 种算子：`Add / Cast / Concat / Constant / Div / Equal / Erf / Expand / Gather /
  Identity / LayerNormalization / MatMul / Mul / Reshape / Shape / Slice / Softmax /
  Split / Sqrt / Transpose / Trilu / Unsqueeze / Where`
- **全部是 TensorRT 原生支持的算子**，无需 plugin
- `onnxruntime` CPU 可加载，动态轴真实生效：`(1,8) (2,8) (1,16) (3,5)` 都能跑
- 数值正确：与 eager 相比 `max|d| ≈ 2.4e-7 ~ 3.6e-7`（float32 量级）

即「ONNX → TensorRT engine」这条常规路径的前置条件是好的。

### 关键约束：导出强制关闭 KV cache

`src/llm/export/_wrapper.py` 的 `ExportCacheWrapper` docstring 明写：

> Forces `use_cache=False` (avoiding KV-cache tracer branching)

`onnx.py` 与 `torchscript.py` 都经过它。后果是**导出的图只能做全量重算**：解码一个
token 要把整个前缀重新跑一遍。以 prompt=8、新增 16 token 为例：

|                      | 处理的位置数 |
| -------------------- | ------------ |
| 无 cache（导出路径） | **248**      |
| 有 cache（eager）    | **24**       |
| 比值                 | **10.3×**    |

这个倍数随生成长度线性增长：新增 `N` 个 token 时，无 cache 是 `O((P+N)²)`，有 cache
是 `O(P+N)`。ADR-004 已经为 serving 做了 paged KV cache；导出路径完全拿不到这个收益。

### Core ML 的额外约束

- `coremltools` 未安装，且 Core ML **只支持 CPU / Apple Silicon**，本机是 A100 ——
  **[INFERENCE]** 无法在本机验证任何 Core ML 产物，只能做形状/算子层面的静态检查。
- Core ML 的 NLP 支持偏向 transformer 的固定 shape 图；动态轴支持有限。

## Decision

**不做 TensorRT / Core ML 的具体实现，本 ADR 记录判断依据与前置条件。**

理由不是「难」，而是投入产出比不明：两项都属于「换运行后端」，而当前后端缺口不在于
算子覆盖（ONNX 23 个算子全支持），而在于**导出路径丢掉了 KV cache**。在没有 cache
的情况下做的 TensorRT 优化，会让一个「10.3 倍冗余计算」的图跑得更快一点——这是在优化
错误的目标。

### 如果要做，顺序必须是

1. **先让导出路径支持 KV cache**（真正的瓶颈，且不依赖任何新依赖）。
   `ExportCacheWrapper` 强制 `use_cache=False` 是因为 trace 时 cache 更新会引入分支。
   可行的方向是把 cache 作为显式输入/输出张量而不是原地可变对象，让 trace 看到一个
   纯函数；这与 ADR-004 的 paged cache 是两套契约，需要单独设计。
2. **再谈 TensorRT**。前置条件已具备（算子全覆盖、opset 17、cu130 wheel 存在），
   主要工作量在 engine 构建 / profile 管理（动态 shape 需要 `optimization_profile`）。
3. **Core ML 单独评估**，且必须在 Apple 硬件上验证；本机做不了。

### 明确不做的事

- 不为了「ROADMAP 有这一项」而先做 TensorRT。先做的收益会被 no-cache 的 10.3 倍吃回去。
- 不在本 ADR 里承诺时间表。这两个都是「换后端」级别的改动，与 Stage-3（ADR-015）
  同属跨架构边界的工作。

## Consequences

### Positive

- 「ONNX 产物本身是否合格」这个最容易被怀疑的环节被实测排除了：算子全支持、动态轴
  有效、数值误差在 float32 量级。以后要做 TensorRT，不必再查这一步。
- 把真正的瓶颈（导出丢 KV cache）与「装个 TensorRT」区分开了，避免在错误目标上优化。
- 记录了两个环境事实，省掉重复探测：`onnxruntime` 没有 CUDA provider；本机装不了
  Core ML 所需平台。

### Negative

- 两项 ROADMAP 依然未做；本 ADR 只提供了决策依据，没有可运行的产物。
- TensorRT 的运行时行为**没有被实测**（安装未完成），本 ADR 的 TensorRT 部分依赖
  `[INFERENCE]`。真要做时必须先补这一步。

### Neutral

- ROADMAP 那两行应当补上指向本 ADR 的说明，避免下一个人以为它们只是「还没排期」。

## References

- ROADMAP §14.2 模型导出 — 「支持 TensorRT 优化」「探索 Core ML 支持 (iOS 部署)」
- `src/llm/export/onnx.py:91` — `export_to_onnx`（opset 17、`dynamo=False`）
- `src/llm/export/_wrapper.py:21` — `ExportCacheWrapper`（强制 `use_cache=False`）
- `src/llm/export/torchscript.py:96` — 同一个 wrapper
- [ADR-004: Paged Attention Serving](004-paged-attention-serving.md) — serving 侧的
  KV cache 契约，导出路径目前用不上
- [ADR-015: ZeRO Stage-3](015-zero-stage-3-parameter-sharding.md) — 同属跨架构边界、
  需先设计再实现的类别
