# 009. AWQ Integration

Date: 2026-08-04

## Status

Accepted

## Context

[ADR-007](./007-gptq-integration.md) 引入了 GPTQ（Hessian-aware 4-bit / 8-bit 量化）。GPTQ 用
二阶信息逐列做误差补偿，质量好但**校准开销大**（Hessian 累积 + Cholesky 求逆 + 逐列更新），
且对校准数据分布的敏感性高。生产部署的另一条主流路线是 **AWQ**（Lin et al., MLSys 2024）：

1. **洞察**：LLM 权重中约 1% 的 "salient channels"（激活幅度大的输入通道）主导量化损失——
   它们的权重相对所在 group 的 max 很小，均匀 group 量化对它们来说是粗粒度
2. **做法**：为每个输入通道搜索一个缩放因子 `s`（2 的幂网格），把 salient 通道的权重**放大**
   后量化，再在 forward 时对输入做 `x / s` 补偿——数学上等价于更细的量化网格，无需保留 FP16 通道
3. **开销**：只依赖**激活均值**（mean abs activation per channel）+ 网格搜索，无 Hessian、
   无逐列误差传播，校准成本远低于 GPTQ

ROADMAP §13.3 明确 "AWQ / SmoothQuant / GGUF 留待后续切片"，本 ADR 记录 AWQ 的集成设计。

## Decision

新增 AWQ 路径，与 GPTQ 路径**正交共存**（`src/llm/quantization/awq.py` +
`_awq_layer.py`），复用同一套校准基础设施与存储约定：

### 架构

```text
Layer 1: _awq_layer.py
  - AWQQuantizedLinear: 打包存储（复用 _gptq_layer 的 4/8-bit 打包约定）
    + input_scales buffer（每输入通道 fp16 缩放，forward 做 x / s 补偿）

Layer 2: awq.py
  - AWQConfig: bits / group_size / sym / n_grid / clip_ratio / layer_policies
  - AWQQuantizer: 逐层处理器，add_batch 累积 per-channel mean-abs 激活
  - _search_scale: 2 的幂网格 + 组内坐标贪心，评估用打包时完全一致的 group 量化
  - quantize_model_awq / quantize_model_awq_with_collector: 模型级入口
```

### 关键设计

1. **组因子化的网格搜索**：group 量化沿输入维解耦——每个 (output row, group) 的 scale 只依赖
   该 group 的列。搜索按 group 独立做坐标贪心：对每个输入通道尝试网格候选（默认 20 个，
   2 的幂、以 1 为中心），用**打包时完全相同的 group 量化器**评估整个 group 的
   activation-weighted 重建误差，保留最优。3 次 pass 让被放大的新 group max 有机会被
   重新评估。激活均值给 salient 通道更高权重——放大它带来的 group 误差上升会被
   activation 加权惩罚，从而得到真实的 trade-off
2. **方向正确性**：salient 通道的缩放方向是 `s > 1`（权重放大、输入缩小）。推导：
   原始空间量化误差 = group step / (2·s)，放大后该通道相对 group max 变大、量化网格更细；
   而 `s < 1` 只会让误差变大，且 uniform ratio 网格是 scale-invariant 的退化情况
3. **诚实评估**：搜索内部用的 `_group_quantize_dequant` 与打包路径 `_pack_weights` 共享
   完全相同的 per-row group-max 数学，搜索结果对最终 layer 的行为是真实的
   （有测试 `test_awq_search_matches_packed_dequantization` 钉住）
4. **运行时补偿在 layer 内**：v1 把 `input_scales` 存进 layer、forward 做 `x / s`——
   数学上精确，无需图分析。跨层 folding（把 `1/s` 折入前一层权重，去掉运行时除法）是
   明确的 follow-up 优化（AutoAWQ 的做法），本切片不做
5. **可选 clipping**：`clip_ratio` 在搜索前做 min-max 裁剪抑制 outlier；默认 None
6. **复用 policy**：`LayerQuantPolicy` + `resolve_layer_policies`（ADR-008）直接支持
   AWQ 的 per-layer 混合精度，无需新机制
7. **与 GPTQ 的关系**：两种算法独立实现、独立 `quantize_model_*` 入口、独立 layer 类型；
   `LayerQuantPolicy` 的 4 字段公共子集是唯一共享抽象

### 用法

```python
from llm.quantization import AWQConfig, quantize_model_awq

# 4-bit group-128 AWQ（默认）
quantize_model_awq(model, calib_iter, AWQConfig())

# 8-bit per-channel + 每层 policy（attention 8-bit, MLP 4-bit）
quantize_model_awq(
    model,
    calib_iter,
    AWQConfig(bits=4, layer_policies=(LayerQuantPolicy(("attn.qkv",), bits=8),)),
)
```

## Alternatives

### Alternative A — 复用 `GPTQQuantizedLinear`，仅加 input_scales buffer

- 优点：少一个 layer 类
- 缺点：类型检查无法区分 AWQ/GPTQ 层（"already quantized" 判定、导出、反量化路径都会
  混淆）；`zeros` buffer 语义不同
- 拒绝原因：独立 layer 类型与 repo 每算法一模块的风格一致，接口更清晰

### Alternative B — 官方实现风格的 uniform-ratio 网格（20 次全矩阵量化）

- 优点：与 llm-awq 代码字面对齐
- 缺点：uniform ratio 在 group-max 重算下是 scale-invariant 的退化搜索（每个候选结果相同）；
  且全矩阵量化 20 次每层，搜索成本 O(n_grid·out_f·in_f) 与组内贪心同阶但语义不成立
- 拒绝原因：退化网格无法通过 "AWQ 优于 naive RTN" 的行为测试；本切片选择数学上成立、
  可验证的组内坐标贪心

### Alternative C — 跨层 folding（AutoAWQ 风格）

- 优点：推理零额外运行时开销
- 缺点：需要前驱层图分析（残差结构下输入来源不唯一）；实现复杂、易错
- 拒绝原因：v1 在 layer 内做精确补偿，folding 作为 follow-up 优化，接口不变

## Consequences

- **正确性收益**：行为测试证明——salient 通道场景下 AWQ 的 activation-weighted 重建误差
  显著低于 naive RTN（约 2.5x），uniform 激活下不劣化
- **校准开销**：只累积 mean-abs 激活（O(in_f) 内存），无 Hessian；网格搜索成本
  O(passes·in_f·n_grid·out_f·group_size)，大模型上需要 GPU 或分块（性能 follow-up）
- **存储**：与 GPTQ 相同的打包布局（4-bit 每字节两个权重），可直接复用评估 / 导出路径
- **运行时**：每个 quantized Linear 多一次逐通道除法（fp16 精度损失可忽略）；folding
  follow-up 可去掉
- **零破坏性变更**：GPTQ / simple-PTQ 路径不动；`LayerQuantPolicy` 是加性复用

## References

- [ADR-007](./007-gptq-integration.md) — GPTQ integration（基础量化路径）
- [ADR-008](./008-mixed-precision-quantization.md) — LayerQuantPolicy 混合精度（本切片复用）
- ROADMAP §13.3 — 高级量化技术（AWQ 原为留待后续切片）
- [Lin et al. 2024, "AWQ: Activation-aware Weight Quantization for LLM
  Compression and Acceleration"](https://arxiv.org/abs/2306.00978) — base algorithm
- Industry references: [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq),
  [AutoAWQ](https://github.com/casper-hansen/AutoAWQ), [vLLM AWQ](https://github.com/vllm-project/vllm)
