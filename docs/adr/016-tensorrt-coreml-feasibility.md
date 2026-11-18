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
   `ExportCacheWrapper` 强制 `use_cache=False` 是因为 cache 更新会引入分支：现行实现
   在 `mha.py:290` 用 `kv_cache.update(k, v)` **原地写** buffer，tracer 无法把「写」
   记进图里。

   **该方向已实测可行**（本 ADR 定稿后补做）：

   | 验证项                                                                        | 结果                                            |
   | ----------------------------------------------------------------------------- | ----------------------------------------------- |
   | 函数式 cache（补齐后的 K/V 作为显式输入/输出，用 `cat` 代替原地写）能否 trace | 通过，产物 21.4 KiB                             |
   | trace 出的图能否**增量**运行（把上一步的 present 喂回下一步的 past）          | 通过，cache 从 `(2,4,0,8)` 逐步长到 `(2,4,3,8)` |
   | 与 eager-with-cache 的数值是否一致                                            | `max|d| = 1.788e-07`（float32 量级）            |

   即「把 cache 由可变对象改成显式张量对」在 trace 层没有障碍。剩余工作是把它接进
   `DecoderModel` 与导出器：需要一个可选入口让模型以函数式契约运行，同时不能破坏
   serving 侧 `KVCache` / `PagedKVCache` 的原地语义（ADR-004）。这是**接口设计**问题，
   不是可行性问题。

   注意：初版验证脚本曾报 `max|d| = 1.06` 的"不匹配"，原因是我在两段代码里各自
   `manual_seed` 但随机数消耗顺序不同——改用**共享同一组输入张量**后误差降到 1.8e-7。
   记录在此，避免下次把脚本缺陷误判为实现缺陷。

   **进展（第一步已落地一部分）**：`src/llm/core/kv_cache.py` 新增
   `FunctionalKVCache`——同样的 `update` 接口，但用 `cat` 构造结果而非原地写，因此
   `mha.py` 一行没改就能用。与 `KVCache` 在真实模型上 **bit-exact 一致**
   （prompt + 3 个 decode step，四个步骤全部 `0.000e+00`）。

   **已完成（第一步落地）**：`src/llm/export/onnx.py` 新增
   `export_decoder_for_generation()`，导出**两张图**（`prefill.onnx` + `decode.onnx`）。
   端到端验证：prefill 一次 + 4 个 decode step，逐步误差

   | step        | max\|eager − onnx\| |
   | ----------- | ------------------- |
   | 0 (prefill) | 4.768e-07           |
   | 1           | 2.384e-07           |
   | 2           | 2.384e-07           |
   | 3           | 2.384e-07           |
   | 4           | 2.384e-07           |

   cache 正确累积到 8 个位置。

   **为什么必须两张图**（这是排查出的关键事实）：`DecoderModel.forward` 的
   `start_pos` 与 `has_past` 都来自 `kv_caches[0].seq_len`，是 **Python 值**，tracer
   会把它们连同 trace 时的形状一起固化成常量。单独看「`is_causal` 被固化」并不致命
   —— 实测一张按 **decode 形状** trace 的图，数值是对的（4.8e-7）。真正致命的是
   **拿 prefill 形状 trace 的图去解码**：解码步骤偏差 ~0.34。因此按各自阶段的形状
   各导一张，是标准 ONNX/TensorRT decoder 做法，也是本函数的实现方式。

   排查中试过但**不需要**的两条路，记录以免重走：
    - 显式 additive mask：rank-2 会撞 SDPA 广播错误
   （`[1,2,4,4]` vs `[4,2,4,4]`），**rank-4 `[1,1,Sq,Sk]` 可用**且与 `has_past`
   路径 bit-exact（`0.000e+00`），但它并不能单独修好单图方案；
    - 显式 `position_ids`：与内部推导 `start_pos` bit-exact（`0.000e+00`），同样不改变
   单图方案的结论。
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
