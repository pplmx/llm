# Mixed-Precision GPTQ - 设计文档

**Date**: 2026-07-22
**Author**: AI Assistant
**Status**: Draft → Pending Review
**Roadmap**: 阶段十三 量化与压缩 §13.1 (Post-Training Quantization - Mixed Precision)
**Related**: ADR-008 (this slice), ADR-007 (GPTQ integration, foundation)

## 目标

在 GPTQ 基础上提供 per-layer 精度分配能力：**单模型内可同时存在 4-bit 与 8-bit 层**，由用户显式声明（user-driven dispatch）。**核心承诺**：

- 零破坏性变更（默认行为完全保留）
- 算法无关 policy 抽象（未来 AWQ / SmoothQuant / QAT 直接复用）
- 与现有 `target_modules` arg 完全正交（filter vs dispatch 各司其职）
- 构造即 fail-fast（policy / config / resolve 三层校验）

## 设计原则

1. **原子 policy** — 每个 `LayerQuantPolicy` 是一个完整的 (target, override bundle) 二元组；不发明"跨 dict 协调"
2. **显式继承** — `None` 字段 = 继承 base config，不靠"字段不存在"的隐式语义
3. **算法无关** — `LayerQuantPolicy` 不引用 GPTQ 概念，字段是所有 PTQ 类算法的公共子集
4. **正交解耦** — 函数级 `target_modules` (filter) ≠ `layer_policies` (dispatch)；互不覆盖
5. **构造时校验** — 所有错误在 `__post_init__` / `resolve` 阶段抛出，runtime 不抛意外
6. **泛型 helper** — `resolve_layer_policies` 接受任意 dataclass，future-proof

## 现有能力（不动的部分）

- `src/llm/quantization/gptq.py` — `GPTQConfig` / `GPTQQuantizer` / `GPTQQuantizedLinear` / `quantize_model_gptq` / `quantize_model_with_collector`
- `src/llm/quantization/_gptq_layer.py` — packed 4-bit / 8-bit storage，per-layer `bits` / `group_size` 已支持
- `tests/quantization/test_gptq_*.py` — 现有 65 tests 全部保留并保持绿色

## 架构总览（三层）

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1 — Policy 定义层   (src/llm/quantization/_policy.py) │
│  ┌────────────────────────────────────────────────────┐    │
│  │  LayerQuantPolicy  (frozen dataclass)              │    │
│  │  - 原子 (target_modules, override_bundle)          │    │
│  │  - 算法无关: bits/group_size/sym/act_order         │    │
│  │  - None = 显式继承                                 │    │
│  └────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────┐    │
│  │  resolve_layer_policies()  (generic helper)        │    │
│  │  - 输入 policies + available_names + base_config   │    │
│  │  - 输出 dict[layer_name, effective_config]         │    │
│  │  - 在此处做跨 policy 校验                          │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                          ↓ 使用
┌─────────────────────────────────────────────────────────────┐
│  Layer 2 — Config 扩展层   (src/llm/quantization/gptq.py)  │
│  GPTQConfig 加 1 个字段:                                    │
│    layer_policies: tuple[LayerQuantPolicy, ...] = ()        │
│  __post_init__ 校验每个元素是 LayerQuantPolicy 实例          │
└─────────────────────────────────────────────────────────────┘
                          ↓ 使用
┌─────────────────────────────────────────────────────────────┐
│  Layer 3 — 算法内 dispatch 层  (src/llm/quantization/gptq.py)│
│  quantize_model_gptq:                                       │
│    - 现有 target_modules arg 仍为 "选层 filter"             │
│    - 新增: 调 resolve_layer_policies 拿 effective_map       │
│    - 每层 effective_map.get(name, config) → dispatch        │
│    - 现有 _quantize_linear_with_gptq 签名不动               │
└─────────────────────────────────────────────────────────────┘
```

## 组件接口

### `LayerQuantPolicy` — 新建，算法无关

```python
@dataclass(frozen=True)
class LayerQuantPolicy:
    """Atomic per-layer quantization override policy. Algorithm-agnostic.

    A LayerQuantPolicy binds a set of target layer names to a bundle of
    override fields. Fields set to None mean "inherit from the algorithm's
    base config". Multiple LayerQuantPolicy in a config are additive; each
    target module must appear in at most one policy (overlap raises
    ValueError at resolve time).
    """

    target_modules: tuple[str, ...]
    bits: int | None = None  # None = inherit; else 4 or 8
    group_size: int | None = None  # None = inherit; else -1 or positive
    sym: bool | None = None  # None = inherit; else True/False
    act_order: bool | None = None  # None = inherit; else True/False

    def __post_init__(self):
        # 字段级校验: target_modules 非空无重复, bits ∈ {4,8},
        # group_size ∈ {-1, >0}, sym/act_order 是 bool 或 None
        ...
```

### `resolve_layer_policies` — 新建 helper（泛型）

```python
def resolve_layer_policies(
    policies: tuple[LayerQuantPolicy, ...],
    available_names: set[str],
    base_config: T,  # generic over any dataclass with override fields
) -> dict[str, T]:
    """Build layer-name -> effective config map from policies.

    Generic over the base config type. Works for GPTQConfig today and any
    future algorithm config (AWQConfig, SmoothQuantConfig, ...) as long as
    the base config dataclass has the four override fields.
    """
    # Phase 1: validate targets exist
    # Phase 2: detect cross-policy overlaps (fail-fast)
    # Phase 3: build effective configs via dataclasses.replace
    ...
```

### `GPTQConfig` — 加 1 个字段

```python
@dataclass(frozen=True)
class GPTQConfig:
    bits: int = 4
    group_size: int = 128
    sym: bool = True
    percdamp: float = 0.01
    blocksize: int = 128
    act_order: bool = False
    static_groups: bool = False

    # NEW: atomic per-layer override policies (additive; empty = no override)
    layer_policies: tuple[LayerQuantPolicy, ...] = ()

    def __post_init__(self):
        # ... 现有 6 个校验不变 ...
        # NEW: validate layer_policies is a tuple of LayerQuantPolicy
        ...
```

### `quantize_model_gptq` — 加 1 处 dispatch

```python
def quantize_model_gptq(model, calib_iter, config=None, target_modules=None, device=None):
    # ... 现有 setup: linear_layers / targets / captured ...

    # NEW: resolve per-layer effective configs (after target_modules filter)
    available_layer_names = {n for n, _ in targets}
    effective_configs = resolve_layer_policies(config.layer_policies, available_layer_names, config)

    # ... 现有 calibration 收集、hook 捕获不变 ...

    # 现有循环，唯一改动: effective_config 替换 config
    for name, layer in targets:
        effective_config = effective_configs.get(name, config)
        new_layer = _quantize_linear_with_gptq(layer, captured[name], effective_config)
        # ... 替换 ...
        logger.info(f"Quantized layer {name}: ... → {bits}-bit, group_size={gs}")
```

## 数据流（成功路径）

```
用户:
  config = GPTQConfig(
      bits=4,
      layer_policies=(
          LayerQuantPolicy(target_modules=("fc1",), bits=8, group_size=-1),
          LayerQuantPolicy(target_modules=("fc2",), bits=4),
      )
  )
  quantize_model_gptq(model, calib_iter, config)

  ↓

Step 1: GPTQConfig.__post_init__ 校验 fields + 每个 policy 校验自身
Step 2: targets = (target_modules arg filter) → {"fc1", "fc2"}
Step 3: resolve_layer_policies → effective_map = {
    "fc1": GPTQConfig(bits=8, group_size=-1, sym=True, ...),
    "fc2": GPTQConfig(bits=4, group_size=128, sym=True, ...),
}
Step 4: per-layer 量化 → model.fc1 是 GPTQQuantizedLinear(bits=8, ...),
                         model.fc2 是 GPTQQuantizedLinear(bits=4, ...)
```

## 错误路径（11 条全部 fail-fast）

| # | 触发 | 抛错位置 | 消息 |
|---|---|---|---|
| E1 | `target_modules=()` | `LayerQuantPolicy.__post_init__` | `target_modules cannot be empty; specify at least one layer name.` |
| E2 | `target_modules=("fc1","fc1")` | 同上 | `target_modules has duplicates within a single policy: ['fc1'].` |
| E3 | `bits=3` | 同上 | `bits must be 4, 8, or None (inherit); got 3.` |
| E4 | `bits=16` | 同上 | 同上 (got 16) |
| E5 | `group_size=0` | 同上 | `group_size must be -1 (per-channel) or positive; got 0.` |
| E6 | `group_size=-128` | 同上 | 同上 (got -128) |
| E7 | `group_size="128"` | 同上 | `group_size must be int or None; got str.` |
| E8 | `layer_policies=("fc1",)` 传字符串 | `GPTQConfig.__post_init__` | `layer_policies[0] must be LayerQuantPolicy; got str.` |
| E9 | policy target 不存在 | `resolve_layer_policies` Phase 1 | `LayerQuantPolicy[0].target_modules ['fc_xxx'] not found in available layers. Available: ['fc1', 'fc2', ...].` |
| E10 | 跨 policy 重叠 | `resolve_layer_policies` Phase 2 | `LayerQuantPolicy.target_modules overlap detected across policies: ['fc1']. Each layer name must appear in at most one policy.` |
| E11 | policy target 在 `target_modules` arg 外 | 同 E9 | 同 E9 (available_names 是 post-filter 集合) |

## 边界场景（**不**抛错的合法行为）

| 场景 | 行为 |
|---|---|
| `GPTQConfig()` 无 layer_policies | 全部用 base config → **与现状完全一致** |
| `LayerQuantPolicy(target_modules=("fc1",), bits=8)` fc1 不在 target_modules arg | E9 抛错（严格模式） |
| Policy 全 None | effective config ≡ base config（冗余但不报错） |
| `group_size=-1` override 成功 | `effective_config.group_size=-1` (per-channel) |
| fp32 字段（percdamp/blocksize/static_groups） | policy 不支持这些字段，**故意不报错也不生效** |

## 顺序保证

校验顺序（用户拿到的总是最先触发的那条）：

1. `LayerQuantPolicy.__post_init__`（E1-E7）— policy 字段级
2. `GPTQConfig.__post_init__`（E8）— config 字段级
3. `resolve_layer_policies` Phase 1（E9, E11）— target 存在性
4. `resolve_layer_policies` Phase 2（E10）— target 跨 policy 重叠

构造时就错的优先级高于运行时错。

## 测试策略

5 个层级，共新增 35 tests，零修改现有 65 tests。

### Level 1 — `LayerQuantPolicy` 单元 (8 tests)

- T1.1-T1.8: 构造、最小化、全字段、frozen、empty/duplicate/bad-bits/bad-group_size/bad-bool

### Level 2 — `resolve_layer_policies` 单元 (12 tests)

- T2.1-T2.12: empty、单 policy 各字段 override、多 policy、unknown target、cross-policy overlap、inherit、strips layer_policies、泛型证明

### Level 3 — GPTQConfig 集成 (5 tests)

- T3.1-T3.5: default、empty、valid、type-check、frozen

### Level 4 — GPTQ mixed-precision 集成 (10 tests)

- T4.1-T4.10: 单层 bits/group_size override、full mixed dispatch、inherit、target_modules 正交、strict mode、log、forward、collector 路径、零行为变化（**关键回归保护**）

### Level 5 — e2e + 文档 (2 tests)

- T5.1-T5.2: TwoLayerMLP mixed e2e + mixed vs uniform memory 对比

### 覆盖率目标

| 模块 | 目标 |
|---|---|
| `_policy.py` | 100% line coverage |
| `gptq.py` 新增行 | 100% line coverage |
| 现有 65 tests | 全绿（不回归） |
| `make ruff` + `make ty` | 零警告零错误 |

## 文件变更清单

```
新增 (3 files)
─────────────────────────────────────────────────────────────
src/llm/quantization/_policy.py                       ~110 行
tests/quantization/test_layer_policy.py               ~250 行
tests/quantization/test_gptq_mixed_precision.py       ~280 行

修改 (4 files, 增量小)
─────────────────────────────────────────────────────────────
src/llm/quantization/gptq.py                          +15 行
src/llm/quantization/__init__.py                      +3 行
docs/adr/008-mixed-precision-quantization.md          新文件 (ADR)
docs/adr/README.md                                    +1 行

文档同步 (2 files)
─────────────────────────────────────────────────────────────
CHANGELOG.md                                          +1 条
ROADMAP.md                                            §13.1 checkbox
```

## Commit 计划（TDD 节奏）

```
commit 1: docs(spec): add mixed-precision quantization design spec
commit 2: docs(adr): ADR-008 mixed-precision GPTQ
commit 3: feat(quant): add LayerQuantPolicy + resolve_layer_policies
         (TDD: RED 20 tests → GREEN 实现 → IMPROVE)
commit 4: feat(quant): wire layer_policies into GPTQConfig + dispatch
         (TDD: RED 15 tests → GREEN 修改 → IMPROVE)
commit 5: docs(quant): sync CHANGELOG + ROADMAP
commit 6: test(quant): ruff + ty + final regression
```

## 不引入的内容（YAGNI）

| 不引入 | 理由 |
|---|---|
| 新 `QuantPolicy` 顶层封装 | 今天没第二个算法用到，加一层间接 |
| Glob pattern (`*.attn.*`) | v1 严格 dotted 命名，与现有 target_modules 风格一致 |
| 自动 heuristic（按激活分布分配 bits） | 这是另一类问题（类似 AWQ saliency），今天只做 user-driven |
| per-layer `percdamp / blocksize / static_groups` | 实际场景罕需 per-layer |

## 风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| 现有 65 tests 隐式依赖 layer_policies 缺省值 | 极低 | commit 4 全量回归 |
| T4.10 输出字节级一致在随机种子下脆弱 | 中 | `torch.manual_seed(42)` + 同一 calib_iter，对比 `weight_packed` bytes |
| LayerQuantPolicy 字段不全导致未来 AWQ 重构 | 低 | T2.12 泛型证明 + `dataclasses.replace` 兼容 |
| 用户误传 target_modules 之外的政策 target | 中 | T4.6 覆盖 → ValueError 引导修正 |

## 时间估算

| 阶段 | 工作量 |
|---|---|
| spec + ADR 撰写 | 1 hour |
| commit 3 (policy + 20 tests) | 2-3 hours |
| commit 4 (GPTQ + 15 tests) | 2-3 hours |
| commit 5 (文档同步) | 30 min |
| commit 6 (全量验证) | 30 min |
| **总计** | **~7-8 hours (1 个工作日)** |

## 验收清单

**功能**：
- [ ] `LayerQuantPolicy` 4 个字段均可独立 override
- [ ] `bits_per_layer={"fc1": 8}` 等价表达可通过 `LayerQuantPolicy(target_modules=("fc1",), bits=8)` 实现
- [ ] 同一层被两个 policy 指向 → ValueError (E10)
- [ ] policy 引用不存在层 → ValueError (E9)
- [ ] `layer_policies=()` 完全等价于不设字段（输出字节级一致）

**质量**：
- [ ] `_policy.py` line coverage = 100%
- [ ] `gptq.py` 修改行 line coverage = 100%
- [ ] 现有 65 GPTQ tests 全绿
- [ ] `make ruff` 零警告
- [ ] `make ty` 零错误

**架构**：
- [ ] `LayerQuantPolicy` 不引用 GPTQ 概念
- [ ] `resolve_layer_policies` 泛型对任意 dataclass 有效
- [ ] `target_modules` arg 与 `layer_policies` 互不干扰

## 开放问题

无（5 节设计已闭环）。
