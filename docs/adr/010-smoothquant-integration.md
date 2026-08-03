# 010. SmoothQuant Integration

Date: 2026-08-04

## Status

Accepted

## Context

[ADR-007](./007-gptq-integration.md) 与 [ADR-009](./009-awq-integration.md) 提供了两条
**weight-only** 量化路径（GPTQ 4/8-bit、AWQ 4/8-bit）。但推理部署的另一半需求是
**weight + activation 双 INT8**：激活用 INT8 才能吃到 INT8 GEMM / 张量核加速，
weight-only 方案无法满足（需要 fp16 激活）。

LLM 激活的经典难题是 **outlier 通道**（个别通道幅度远大于其余通道），使 per-tensor
INT8 激活量化的误差爆炸。**SmoothQuant**（Xiao et al., ICML 2023）的解法：

1. 对每个输入通道计算平滑因子 `s_j = max|X_j|^α / max|W_j|^(1−α)`
2. 把难度从激活迁移到权重：权重乘 `s`（`W' = W·s`），激活除 `s`（`x' = x/s`），
   数学上 `y = W'·x' = W·x` 不变
3. 两者都做 INT8 对称量化：权重 per-row，激活 per-tensor

ROADMAP §13.3 的 "SmoothQuant" 原为留待后续切片，本 ADR 记录集成设计。

## Decision

新增 SmoothQuant 路径（`src/llm/quantization/smooth.py` +
`_smooth_layer.py`），与 GPTQ / AWQ / simple-PTQ 正交共存，复用同一套校准基础设施：

### 架构

```text
Layer 1: _smooth_layer.py
  - SmoothQuantLinear: int8 权重（per-row scale）+ 每张量激活 fake-quant
    + input_scales buffer（每输入通道 fp16 平滑因子，forward 做 x / s）

Layer 2: smooth.py
  - SmoothQuantConfig: alpha / search_alpha / bits=8 / sym / layer_policies
  - SmoothQuantQuantizer: add_batch 累积 per-channel max-abs 激活
  - _smoothing_scales / _activation_scale: 平滑因子 + per-tensor 激活 scale
  - _eval_layer_error: alpha 候选的诚实输出误差评估
  - quantize_model_smoothquant / quantize_model_smoothquant_with_collector
```

### 关键设计

1. **统计量即尺度**：平滑因子只依赖 per-channel 激活 max 与权重 max；且 per-tensor
   激活 scale 有闭式解 `max_j(act_max[j]/s_j)/127`（`s` 逐通道常数），**无需保留校准
   批次**即可完成量化——内存 O(in_f)。只有 `search_alpha=True` 才按层保留批次
2. **alpha 搜索**：`alpha` 默认 0.5（论文推荐平衡值）；`search_alpha=True` 时在
   {0.25, 0.5, 0.75, 1.0} 网格上按层选输出重建误差最小的 alpha，评估用与最终 layer
   完全一致的量化数学（诚实搜索）
3. **v1 范围**：SmoothQuant 是 INT8 方法——`bits` 只允许 8；权重 per-channel
   （`group_size=-1`）。策略覆盖走 `LayerQuantPolicy`（ADR-008），非法值由
   `__post_init__` 在 `dataclasses.replace` 时 fail-fast（`replace` 会重跑校验）
4. **运行时补偿在 layer 内**：`input_scales` 存进 layer、forward 做 `x/s`，数学精确、
   无需图分析；跨层 folding（把 `1/s` 折入前一层权重）与真实 INT8 GEMM kernel 是
   明确 follow-up
5. **激活 fake-quant 前置**：forward 对 `x/s` 做 per-tensor round/clamp 到 INT8，
   与权重 dequant 一起在 fp32 里计算——与 GPTQ/AWQ 的 dequant-at-forward 策略一致，
   正确性优先，kernel 优化后续
6. **对称量化**：v1 仅 `sym=True`（无 zero-point），与 GPTQ/AWQ v1 一致；非对称是
   follow-up

### 用法

```python
from llm.quantization import SmoothQuantConfig, quantize_model_smoothquant

# 默认 alpha=0.5, INT8 weight+activation
quantize_model_smoothquant(model, calib_iter, SmoothQuantConfig())

# 按层搜索最优 alpha（保留校准批次，内存更高）
quantize_model_smoothquant(model, calib_iter, SmoothQuantConfig(search_alpha=True))
```

## Alternatives

### Alternative A — 复用 AWQQuantizedLinear 加激活量化

- 优点：少一个 layer 类
- 缺点：AWQ 是 weight-only 4/8-bit 语义（nibble 打包、可选 4-bit），与 SmoothQuant
  的 INT8 权重 + 激活量化语义不同；类型检查无法区分
- 拒绝原因：独立 layer 类型与 repo 每算法一模块的风格一致

### Alternative B — 把激活 scale 也做成 per-channel

- 优点：激活误差更小
- 缺点：失去 INT8 GEMM 的前提（激活 per-channel 需要额外 kernel 支持）；偏离论文
- 拒绝原因：SmoothQuant 的卖点就是 per-tensor 激活 + per-row 权重，保持论文语义

### Alternative C — 跨层 folding + 真 INT8 kernel（vLLM 风格）

- 优点：推理零额外开销、吃到张量核
- 缺点：需要图分析 + 专用 kernel 层，v1 范围过大
- 拒绝原因：先保证算法正确性与可测试性；folding / kernel 作为 follow-up，
  layer 接口不变

## Consequences

- **质量收益**：行为测试证明——激活 outlier 通道场景下 SmoothQuant 的输出重建误差
  比无平滑的 INT8 W+A 量化低约一个数量级；正常激活下不显著劣化；alpha 搜索在网格上
  找到不差于极端的 alpha
- **校准开销**：非搜索路径只累积 per-channel max-abs 激活（O(in_f) 内存），无 Hessian、
  无网格搜索；`search_alpha` 需要保留每层校准批次（与 AWQ 搜索同阶）
- **存储**：权重 int8 直接存储（无打包），per-row scale fp16 + per-tensor act_scale
  fp16 + input_scales fp16
- **运行时**：每层多一次逐通道除法 + 激活 round/clamp；folding + INT8 GEMM follow-up
- **零破坏性变更**：GPTQ / AWQ / simple-PTQ 路径不动；`_single_thread_reductions`
  移入共享 `calibration.py`（行为不变）

## References

- [ADR-007](./007-gptq-integration.md) — GPTQ integration（weight-only 路径）
- [ADR-009](./009-awq-integration.md) — AWQ integration（weight-only 路径）
- [ADR-008](./008-mixed-precision-quantization.md) — LayerQuantPolicy（策略复用）
- ROADMAP §13.3 — 高级量化技术（SmoothQuant 原为留待后续切片）
- [Xiao et al. 2023, "SmoothQuant: Accurate and Efficient Post-Training
  Quantization for Large Language Models"](https://arxiv.org/abs/2211.10438) —
  base algorithm
- Industry references: [mit-han-lab/smoothquant](https://github.com/mit-han-lab/smoothquant),
  [vLLM SmoothQuant](https://github.com/vllm-project/vllm), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
