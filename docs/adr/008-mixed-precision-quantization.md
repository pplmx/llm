# 008. Mixed-Precision GPTQ

Date: 2026-07-22

## Status

Accepted

## Context

[ADR-007](./007-gptq-integration.md) 在 2026-07-22 引入了 GPTQ 路径，提供 Hessian-aware 4-bit / 8-bit 量化（统一精度，单模型所有层共享 `bits` / `group_size`）。

但生产部署的实践表明：**统一精度通常不是最优策略**：

1. **质量敏感层（如 attention Q/K/V 投影）需要更高精度**，否则 perplexity 显著退化
2. **容量敏感层（如 FFN down-proj）可承受更低精度**，节省存储和带宽
3. **业界事实标准**（vLLM / TGI / llama.cpp / AutoGPTQ）均支持 per-layer bit 分配
4. **用户研究反馈**：部署方反复要求"attention 8-bit + MLP 4-bit"组合

直接扩展 `GPTQConfig` 加 `bits_per_layer: dict[str, int]` 是最简路径，但有结构性缺陷：

- `bits_per_layer` 与未来的 `group_size_per_layer` / `sym_per_layer` / `act_order_per_layer` 会变成 N 个平行 dict，**key 一致性必须手动保证**
- 字段语义不明确："字典里没有该层"是"用 base"还是"不量化"？容易踩坑
- 算法绑死 GPTQ 概念（bits/group_size 是 GPTQ 命名），未来 AWQ / SmoothQuant 重新发明一次

## Decision

引入 `LayerQuantPolicy` 原子 policy 抽象 + `GPTQConfig.layer_policies` 字段，提供 per-layer 精度 dispatch：

### 架构

```
Layer 1: _policy.py (算法无关)
  - LayerQuantPolicy (frozen dataclass, 4 字段 + target_modules)
  - resolve_layer_policies (泛型 helper, 输出 dict[layer_name, effective_config])

Layer 2: gptq.py (Config 扩展)
  - GPTQConfig 加 layer_policies: tuple[LayerQuantPolicy, ...] = ()
  - __post_init__ 校验每个元素是 LayerQuantPolicy 实例

Layer 3: gptq.py (Dispatch)
  - quantize_model_gptq 调 resolve_layer_policies + 每层 effective_map.get(name, config)
```

### 关键设计

1. **原子 policy**：每个 `LayerQuantPolicy` 是 `(target_modules, override_bundle)` 二元组，无跨 dict 协调
2. **显式继承**：override 字段为 `None` = 继承 base config（语义唯一）
3. **算法无关**：`LayerQuantPolicy` 在 `_policy.py`，字段 `bits / group_size / sym / act_order` 是所有 PTQ 类算法的公共子集
4. **正交解耦**：函数级 `target_modules` arg = filter；`layer_policies` = dispatch；互不覆盖
5. **泛型 helper**：`resolve_layer_policies` 接受任意 dataclass，future-proof for AWQ / SmoothQuant
6. **构造即 fail-fast**：三层校验（policy 字段 / config 字段 / resolve 阶段），共 11 条错误路径
7. **零破坏性变更**：`layer_policies` 默认空 tuple，54 个现有 GPTQ tests 零修改

### 用法

```python
# 例 1: 全 4-bit（与现状完全一致）
GPTQConfig(bits=4)

# 例 2: 注意力 8-bit + MLP 4-bit（最常见 mixed-precision）
GPTQConfig(
    bits=4,
    layer_policies=(
        LayerQuantPolicy(target_modules=("model.layers.*.self_attn.*",), bits=8),
        LayerQuantPolicy(target_modules=("model.layers.*.mlp.*",), bits=4),
    ),
)

# 例 3: 只覆盖 bits，group_size 继承
LayerQuantPolicy(target_modules=("lm_head",), bits=8)  # group_size=None → 继承 128
```

### 实施切片

- 3 个新文件（`_policy.py` + 2 个 test 文件），4 个文件改动（gptq.py / __init__.py / ADR / README）
- ~120 行代码净增量，35 个新 tests
- ~7-8 小时工作量，1 个工作日
- TDD 节奏：每个 commit RED → GREEN → IMPROVE

## Consequences

### Positive

- **生产价值兑现**：用户可按"质量敏感 vs 容量敏感"自由分配精度，典型场景 attention 8-bit + MLP 4-bit，avg ≈ 4.5-bit 但 perplexity ≈ FP
- **零回归**：`layer_policies=()` 与不设字段字节级一致，54 个现有 GPTQ tests 不动
- **未来算法免费**：AWQ / SmoothQuant / QAT 集成时直接复用 `LayerQuantPolicy` + `resolve_layer_policies`
- **API 友好**：用户不需要手动协调多个 dict，atomic policy 自带 (target, config) 二元组语义
- **错误信号清晰**：11 条错误路径每条都有精确 message + 可用层名样本

### Negative

- **多 1 个文件 / 1 个公共类**：`_policy.py` + `LayerQuantPolicy` 增加 API 表面（但有明确职责）
- **API 略复杂**：用户从 `bits_per_layer={"fc1": 8}` 改为 `layer_policies=(LayerQuantPolicy(target_modules=("fc1",), bits=8),)`，多一层嵌套
- **泛型 helper 调试难度**：T 泛型无运行时校验，IDE 类型提示可能不够友好（已通过 T2.12 测试证明）
- **不支持 glob pattern**（v1）：strict dotted 命名与现有 `target_modules` 风格一致，但未来需要 glob 时要加新特性

### Neutral

- **`GPTQQuantizedLinear` 零改动**：已经支持 per-layer `bits` / `group_size` 字段
- **`quantize_model_with_collector` 零改动**：薄包装自动继承 mixed-precision 能力
- **日志格式微变**：从 `→ 4-bit packed` 改为 `→ 4-bit, group_size=128`（更明确）

## Alternatives Considered

### Alternative A — 简单 `bits_per_layer` dict

```python
@dataclass(frozen=True)
class GPTQConfig:
    bits: int = 4
    bits_per_layer: dict[str, int] | None = None  # NEW
    group_size_per_layer: dict[str, int] | None = None  # NEW (future)
```

- 优点：API 表面最小，2 行代码改动
- 缺点：N 个平行 dict 必须手动同步 key；字段语义模糊；算法绑死 GPTQ
- 拒绝原因：未来加 `sym_per_layer` / `act_order_per_layer` 时 N 翻倍；AWQ 重新发明

### Alternative B — per-layer 完整 `GPTQConfig` map

```python
layer_configs: dict[str, GPTQConfig] | None = None
```

- 优点：完全灵活，每层可独立选择 sym / act_order / percdamp / blocksize
- 缺点：用户要为每个层构造完整 dataclass，99% 场景下只想改 bits/group_size
- 拒绝原因：API 重量级，实际场景过度设计

### Alternative C — `QuantPolicy` 顶层封装

```python
@dataclass(frozen=True)
class QuantPolicy:
    default: GPTQConfig
    overrides: dict[str, GPTQConfig]

quantize_model_gptq(model, calib, policy=QuantPolicy(default=base, overrides={...}))
```

- 优点：与未来"policy 可跨算法"的设想对齐
- 缺点：今天没第二个算法用到，加一层间接
- 拒绝原因：YAGNI；今天先做"算法无关 policy 抽象"，policy 顶层封装是未来切片

### Alternative D — Glob pattern + 字符串配置

```python
GPTQConfig(bits=4, layer_glob_overrides={
    "*.attn.*": {"bits": 8},
    "*.mlp.*": {"bits": 4},
})
```

- 优点：用户更直观
- 缺点：glob 匹配逻辑复杂；与现有 strict `target_modules` 风格不一致
- 拒绝原因：v1 严格 dotted 命名保持简洁；未来按需加 glob 支持

## References

- [ADR-007](./007-gptq-integration.md) — GPTQ integration (foundation)
- [Design spec](../superpowers/specs/2026-07-22-mixed-precision-quantization-design.md) — full 5-section design
- [ROADMAP §13.1](../ROADMAP.md#阶段十三-量化与压缩-) — Post-Training Quantization (mixed precision)
- [Frantar et al. 2022, "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"](https://arxiv.org/abs/2210.17323) — base algorithm
- Industry references: [vLLM per-layer quant](https://github.com/vllm-project/vllm), [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ), [llama.cpp quant types](https://github.com/ggerganov/llama.cpp)
