# GPTQ 集成 - 设计文档

**Date**: 2026-07-22
**Author**: AI Assistant
**Status**: Draft → Pending Review
**Roadmap**: 阶段十三 量化与压缩 §13.3 (集成 GPTQ)

## 目标

在现有 `src/llm/quantization/` 模块中集成 GPTQ（Frantar et al. 2022, "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"），提供生产可用的 4-bit 量化路径。**核心承诺**：真实减少 4× 权重存储 + 量化误差低于 naive round-to-nearest。

## 设计原则

1. **与现有 INT8/INT4 simple-PTQ 正交** — GPTQ 是逐层 Hessian-aware 算法，simple-PTQ 是 post-hoc 直接量化，强行合并会污染两套测试
2. **4-bit 真打包** — 2 weights/byte int8 storage，不做"假装 4-bit"（int8 槽位只 mask）
3. **双重 calibration 入口** — standalone (`Iterator[Tensor]`) + 复用现有 `CalibrationDataCollector`
4. **配置错误 fail-fast** — `GPTQConfig.__post_init__` 拒绝非法组合，不延迟到 forward 才炸
5. **CLI 平行 `llm-migrate-ckpt`** — 用户可直接 `llm-quantize gptq ...`，不写 Python

## 现有能力（不动的部分）

- `src/llm/quantization/ptq.py` — `QuantConfig` + `QuantizedLinear` (INT8 storage, symmetric/asymmetric, per-channel/per-tensor) — **零改动**
- `src/llm/quantization/calibration.py` — `ActivationStats` + `CalibrationDataCollector` — **零改动**（GPTQ 复用其数据采集能力）
- `tests/quantization/test_quantization.py` — 现有 INT8 测试 — **零改动**

## 新增模块

```
src/llm/quantization/
├── __init__.py             ← 扩导出: GPTQConfig, GPTQQuantizedLinear, GPTQQuantizer, quantize_model_gptq, quantize_model_with_collector
├── ptq.py                  ← 不动
├── calibration.py          ← 不动
├── gptq.py                 ← 新增 (GPTQConfig, GPTQQuantizer, 顶层 entry)
└── _gptq_layer.py          ← 新增 (GPTQQuantizedLinear 存储 + packed 4-bit)

src/llm/cli/
└── quantize.py             ← 新增 (llm-quantize gptq subcommand)

tests/quantization/
├── test_quantization.py    ← 不动
├── test_gptq_algorithm.py  ← 新增
├── test_gptq_layer.py      ← 新增
└── test_gptq_end_to_end.py ← 新增

tests/cli/
└── test_quantize_cli.py    ← 新增

docs/adr/
└── 007-gptq-integration.md ← 新增
```

## 公共 API

### 配置

```python
@dataclass(frozen=True)
class GPTQConfig:
    bits: int = 4                          # 4 or 8
    group_size: int = 128                  # -1 = per-channel
    sym: bool = True                       # symmetric quantization
    percdamp: float = 0.01                 # Hessian damping percentage
    blocksize: int = 128                   # column block size for memory control
    act_order: bool = False                # True: sort cols by diag(H) desc
    static_groups: bool = False            # True: share group partition across layers

    def __post_init__(self):
        if self.bits not in (4, 8):
            raise ValueError(f"GPTQConfig.bits must be 4 or 8, got {self.bits}.")
        if self.group_size != -1 and self.group_size < 0:
            raise ValueError(f"group_size must be -1 (per-channel) or positive, got {self.group_size}.")
        if not (0.0 < self.percdamp < 1.0):
            raise ValueError(f"percdamp must be in (0, 1), got {self.percdamp}.")
        if self.blocksize <= 0:
            raise ValueError(f"blocksize must be positive, got {self.blocksize}.")
        if self.group_size > 0 and self.blocksize % self.group_size != 0:
            raise ValueError(
                f"blocksize ({self.blocksize}) must be divisible by group_size ({self.group_size}) "
                f"for correct packing alignment."
            )
```

### 算法层

```python
class GPTQQuantizer:
    """Stateful per-layer GPTQ processor."""

    def __init__(self, layer: nn.Linear, config: GPTQConfig): ...
    def add_batch(self, x: torch.Tensor) -> None:
        """Accumulate Hessian: H += (2/N) · X^T X for the calibration batch."""
    def quantize(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Run GPTQ on accumulated H, return (W_packed_int8, scales_fp16, zeros_int8_or_None)."""
```

### 存储层

```python
class GPTQQuantizedLinear(nn.Module):
    """GPTQ-quantized Linear with packed 4-bit (or 8-bit) weight storage."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        weight_packed: torch.Tensor,     # int8, 2× 4-bit per byte
        scales: torch.Tensor,            # fp16, [out, in/group_size]
        zeros: torch.Tensor | None,      # int8, [out, in/group_size] or None
        bits: int = 4,
        group_size: int = 128,
        sym: bool = True,
    ): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def _unpack_weights(self) -> torch.Tensor:
        """Unpack int8 → fp32. Only called inside forward, never cached."""
    @classmethod
    def from_linear(cls, linear: nn.Linear, ...) -> "GPTQQuantizedLinear": ...
```

### 顶层入口

```python
def quantize_model_gptq(
    model: nn.Module,
    calib_iter: Iterator[torch.Tensor],
    config: GPTQConfig | None = None,
    target_modules: Iterable[str] | None = None,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Standalone entry: user supplies calibration iterator."""

def quantize_model_with_collector(
    model: nn.Module,
    collector: CalibrationDataCollector,
    n_samples: int,
    config: GPTQConfig | None = None,
    target_modules: Iterable[str] | None = None,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Trainer-loop entry: reuse existing CalibrationDataCollector."""
```

### CLI

```
llm-quantize gptq --model PATH --output PATH \
    --calib-data PATH --bits 4 --group-size 128 \
    [--act-order] [--sym] [--percdamp 0.01] [--blocksize 128] \
    [--target-modules q_proj,k_proj,v_proj,o_proj]
```

`--calib-data` 一行一个 raw text 样本，内部用模型默认 tokenizer 处理（standalone 路径，不依赖训练框架）。

## 数据流

```
用户代码                         GPTQQuantizer                    nn.Linear
  │                                   │                               │
  │  model, calib_iter, config        │                               │
  ├──────────────────────────────────►│                               │
  │                                   │  register_forward_hook         │
  │                                   ├──────────────────────────────►│
  │                                   │                               │
  │  for batch in calib_iter:         │                               │
  │  ──────────────────────────────►  │  forward(batch)                │
  │                                   ├──────────────────────────────►│
  │                                   │  hook captures X (input)      │
  │                                   │  hook calls quantizer.add_batch│
  │                                   │  quantizer.H += 2/N · X^T X   │
  │                                   │                               │
  │  ──────── (unhook) ────────────►  │                               │
  │                                   │                               │
  │  for each (name, Linear):         │                               │
  │  ──────────────────────────────►  │  GPTQQuantizer(layer, cfg)    │
  │                                   │  quantize() algorithm:        │
  │                                   │  ─ 1. H_inv = chol_inv(H+λI)  │
  │                                   │  ─ 2. (opt) act-order sort    │
  │                                   │  ─ 3. for j in blocksize:     │
  │                                   │       q(W[:,j]), err, update  │
  │                                   │  ─ 4. for each group g cols:  │
  │                                   │       compute scale, zeros    │
  │                                   │       pack 4-bit to int8      │
  │                                   │  ─ 5. GPTQQuantizedLinear()   │
  │                                   │  ─ 6. _replace_module(parent) │
  │                                   ├──────────────────────────────►│  setattr
  │  <return quantized model>         │                               │
  │◄──────────────────────────────────┤                               │
```

**关键设计**：
- 用 `register_forward_hook`（捕获输入 X）而非 `register_forward_pre_hook`
- 每层**立即 unhook + 释放 X**，避免累积显存（Hessian 累积是 GPTQ 内存峰值点）
- 双重入口最终汇入同一 `GPTQQuantizer.quantize()` 路径

## 错误处理

| 错误类型 | 抛出位置 | 用户行为 |
|---------|---------|---------|
| 配置错误（bits∉{4,8}、group_size 无效、blocksize/group_size 不整除） | `GPTQConfig.__post_init__` | 改 config |
| 形状错误（calib batch 维度不对、layer 已 quantize） | `GPTQQuantizer.add_batch` / `_replace_module` | 修数据 |
| 数值错误（Hessian 奇异、不可逆） | `GPTQQuantizer.quantize` | 抛带 `percdamp` 提示的 RuntimeError |
| 运行时错误（target_modules 未匹配、模型无 Linear） | `quantize_model_gptq` | 检查模型结构 |
| I/O 错误（checkpoint 路径错、权限） | CLI 层 | 检查路径/权限 |

**原则**：
- 配置错误 fail-fast（构造时拒绝）
- 数值错误带 recovery hint（`percdamp` 调大提示），不裸 `RuntimeError`
- 已 quantize 层检测，避免双重量化
- 不静默吞错，所有 try/except 带上下文 + actionable hint

## 测试策略（三层）

### Layer 1 — 算法 (`test_gptq_algorithm.py`)

| 测试 | 验证 |
|------|------|
| `test_hessian_accumulates_correctly` | `add_batch` 多次等价一次性累加 |
| `test_cholesky_inverse_matches_torch` | `U^T U == H^-1` |
| **`test_gptq_lower_error_than_rtn`** | **核心: GPTQ MSE < RTN baseline（守住 GPTQ 优势）** |
| `test_act_order_changes_column_sequence` | 启用 act_order 后按 diag(H) desc 排序 |
| `test_per_channel_vs_group` | `group_size=-1` vs `128` 量化结果形状差异 |
| `test_singular_hessian_recoverable_with_damp` | 0.01 damp 失败, 0.1 damp 成功 |
| `test_zero_calibration_raises_actionable` | H 全零 → 抛 percdamp 提示 |

### Layer 2 — 存储 (`test_gptq_layer.py`)

| 测试 | 验证 |
|------|------|
| `test_pack_unpack_round_trip` | int8 ∈ [-8,7] pack → unpack 还原 |
| **`test_packed_storage_half_size`** | **核心: 4-bit packed bytes == numel / 2** |
| `test_forward_close_to_fp32` | 4-bit 输出 vs fp32 cosine_sim > 0.999, MSE < 1e-3 |
| `@parametrize` `test_forward_correctness_across_group_sizes` | `group_size ∈ {-1, 128}` 都通过 |
| `@parametrize` `test_symmetric_and_asymmetric` | sym + asym 都正确 |
| `test_bias_preserved` / `test_no_bias_path` | bias 完整处理 |

### Layer 3 — 端到端 (`test_gptq_end_to_end.py`)

| 测试 | 验证 |
|------|------|
| `test_small_transformer_quantize_then_forward` | 2 层 Decoder + 100 calib samples, fp32 vs int4 cosine > 0.95 |
| `test_save_load_quantized_model_round_trips` | save_pretrained → load → forward 一致 |
| `test_quantize_does_not_break_forward_contract` | 现有 forward signature 不变 |
| `test_target_modules_filters_correctly` | 只量化指定模块，其他保持 fp32 |

### Layer 4 — CLI (`tests/cli/test_quantize_cli.py`)

| 测试 | 验证 |
|------|------|
| `test_cli_help` | `--help` 不炸 |
| `test_cli_missing_args_exits_nonzero` | 缺参 → 非 0 退出 |
| `test_cli_happy_path` | 完整 args → 生成输出文件 |
| `test_cli_invalid_bits_errors` | `--bits 16` → 报错 |

### 覆盖率目标

| 文件 | 行覆盖目标 |
|------|----------|
| `gptq.py` | 95%+ |
| `_gptq_layer.py` | 90%+ |
| `cli/quantize.py` | 85%+ |

**TDD 工作流**：Layer 1 (算法) → 红灯 → 实现 → 绿灯 → Layer 2 → ... → Layer 3 → e2e

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| 数值精度不足（极端 calibration 数据） | 暴露 `percdamp` + `act_order` 两个旋钮给用户 |
| 4-bit packed unpack 慢于 cached fp32 | `_unpack_weights()` 仅在 forward 调用，不缓存 fp32 weight（避免双份内存） |
| Hessian 计算 OOM（极宽 hidden） | `blocksize` 控制列块，量化完一块立即释放中间变量 |
| 与 PEFT 集成冲突 | GPTQ 不动 PEFT_REGISTRY；PEFT-aware 路径留待后续独立切片 |
| CI 无 GPU 跑大型矩阵 | 所有测试用 `torch.float64` 算 + 32×32~128×128 小矩阵，CPU 几秒可过 |

## 不在本次范围

- AWQ / SmoothQuant 集成（13.3 后续切片，共享 calibration hookup）
- Quantization-Aware Training（QAT, 13.2）
- GPTQ + PEFT 联合路径（PEFT-aware quantization 留独立切片）
- Mixed-precision per-layer sensitivity（13.1 后续）
- TensorRT / Core ML 导出（14.2）

## 验收标准

- [ ] `tests/quantization/test_gptq_*.py` 全部通过（4 个文件，预计 30+ tests）
- [ ] `tests/cli/test_quantize_cli.py` 通过
- [ ] 现有 `tests/quantization/test_quantization.py` 零修改通过（回归保护）
- [ ] `make test` 全绿
- [ ] CLI `llm-quantize gptq --help` 输出正确
- [ ] `docs/adr/007-gptq-integration.md` 提交
- [ ] `CHANGELOG.md` `[Unreleased]` Added 段更新
- [ ] `ROADMAP.md` 阶段十三 §13.3 第一个 checkbox 勾选
