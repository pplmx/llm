---
tags:
  - 指南
  - 量化
---

# 模型量化指南

本文介绍框架内置的后训练量化（PTQ）方法——GPTQ、AWQ、SmoothQuant 和简单 PTQ——
以及如何用 `llm-quantize` CLI 或 Python API 对训练好的模型做压缩。

> 对应 API 文档见 [llm.quantization](../api/quantization.md)，CLI 参数速查见
> [CLI 命令参考](../reference/cli.md#llm-quantize)。

## 方法总览

| 方法           | 权重位宽 | 激活量化 | 需要校准数据 | 适合场景                                     |
| -------------- | -------- | -------- | ------------ | -------------------------------------------- |
| 简单 PTQ (RTN) | INT8/4   | 否       | 否           | 快速基线、无校准数据可用                     |
| GPTQ           | INT4/8   | 否       | 是           | 通用 4-bit 压缩，Hessian 感知（主流默认）    |
| AWQ            | INT4/8   | 否       | 是           | 激活存在显著离群通道时的权重压缩             |
| SmoothQuant    | INT8     | 是       | 是           | 需要 INT8 权重+激活部署（W8A8）的推理路径    |
| Mixed-Precision| 4/8 混合 | -        | 是           | 关键层保留 8-bit、其余 4-bit 的精度/体积权衡 |

共同约定：

- 权重被替换为打包后的量化线性层（`GPTQQuantizedLinear` / `AWQQuantizedLinear` /
  `SmoothQuantLinear`），正向传播在内存中解量化，数学上等价于原始层。
- 所有方法都拒绝重复量化已量化层，并在模型不含 `nn.Linear`、`target_modules`
  无匹配、校准数据为空时给出明确错误。
- 量化后的模型仍是 `DecoderModel`，可直接被 `llm-serve` 的 loader 加载
  （`llm-serve` 自动识别量化 blob）。

## 何时选择哪种方法

1. **先跑简单 PTQ** 作为基线——如果精度损失已经可接受，不需要引入校准数据。
2. **默认生产选 GPTQ 4-bit**（`group_size=128`）：Hessian 感知的误差校正让
   4-bit 也能保持接近 fp16 的质量。
3. 如果 GPTQ 在 4-bit 下精度损失明显且激活分布有离群通道，**换 AWQ** 或
   在关键层上保留 8-bit（mixed-precision）。
4. 如果部署目标要求 **W8A8 INT8 权重+激活**（例如特定推理芯片），用
   **SmoothQuant**。

## 用 CLI 量化（推荐）

```bash
# 原始文本校准（需要 HF tokenizer）
uv run llm-quantize gptq \
    --model checkpoints/epoch_5.pt \
    --output checkpoints/epoch_5-int4.pt \
    --calib-data calibration.txt \
    --tokenizer gpt2 \
    --bits 4 \
    --group-size 128 \
    --act-order

# 预分词校准（无需 --tokenizer）
uv run llm-quantize gptq \
    --model checkpoints/epoch_5.pt \
    --output checkpoints/epoch_5-int8.pt \
    --calib-data-tokens calib_tokens.pt \
    --bits 8
```

退出码约定：`0` 成功；`1` 参数校验失败（`--bits` 非法、缺 `--tokenizer`、
`--model` 不存在等）；`2` 运行期失败（`torch.load` / 分词 / 量化内核 / 保存）。
完整参数表与校验规则见 [CLI 命令参考](../reference/cli.md#llm-quantize)。

## 用 Python API 量化

所有算法共享同一个调用约定：`quantize_model_<algo>(model, calib_iter, config, ...)`。

```python
import torch
from llm.quantization import GPTQConfig, quantize_model_gptq

model = torch.load("checkpoints/epoch_5.pt")  # 或从 Config 重建
calib = [torch.randn(4, 16, 128)]  # 用真实校准数据替换

quantized = quantize_model_gptq(
    model,
    iter(calib),
    config=GPTQConfig(bits=4, group_size=128),
)
torch.save(quantized, "checkpoints/epoch_5-int4.pt")
```

AWQ / SmoothQuant 使用同一模式：

```python
from llm.quantization import AWQConfig, SmoothQuantConfig
from llm.quantization import quantize_model_awq, quantize_model_smoothquant

quantize_model_awq(model, iter(calib), config=AWQConfig(bits=4, group_size=128))
quantize_model_smoothquant(model, iter(calib), config=SmoothQuantConfig(alpha=0.5))
```

### 复用训练期校准数据

训练回调收集的 `CalibrationDataCollector` 可以直接喂给
`quantize_model_with_collector`，避免重新跑一遍校准 forward：

```python
from llm.quantization import quantize_model_with_collector

quantize_model_with_collector(model, collector, n_samples=32)
```

## 混合精度（按层 4/8-bit）

`LayerQuantPolicy` 把 `bits` / `group_size` / `sym` / `act_order` 绑定到指定层，
未指定字段继承算法基配置。policy 目标必须存在于 `target_modules` 过滤后的层集合，
否则在解析时直接报错（fail-fast）。

```python
from llm.quantization import GPTQConfig, LayerQuantPolicy, quantize_model_gptq

config = GPTQConfig(
    bits=4,
    group_size=128,
    layer_policies=(
        LayerQuantPolicy(target_modules=("lm_head",), bits=8),  # 关键层保 8-bit
        LayerQuantPolicy(target_modules=("layers.0.mlp",), bits=4),  # 显式 4-bit
    ),
)
quantize_model_gptq(model, iter(calib), config=config)
```

## 量化质量验证

仓库提供了复现用的困惑度门禁脚本（真实模型 + WikiText-2 校准）：

```bash
uv run python scripts/quantize_eval.py \
    --model checkpoints_wiki/epoch_10 \
    --tokenizer-path /path/to/tokenizer.pt \
    --device cuda:0
```

它会加载 checkpoint → 评估量化前 perplexity → GPTQ 4-bit 量化 → 再评估，
打印前后对比。建议在发布量化模型前以这种真实数据验证，而不是只看单测里的
重建误差。

## 量化模型的服务部署

`llm-serve` 的 loader 能直接识别 `llm-quantize` 输出的量化 blob
（self-contained，打包权重 + scale + 每层量化参数），无需重建模型结构：

```bash
LLM_SERVING_MODEL_PATH=checkpoints/epoch_5-int4.pt \
LLM_SERVING_TOKENIZER_TYPE=hf \
LLM_SERVING_TOKENIZER_PATH=gpt2 \
uv run llm-serve
```

## 进一步阅读

- API：`llm.quantization`（[quantization.md](../api/quantization.md)）
- CLI 参数：[llm-quantize](../reference/cli.md#llm-quantize)
- 架构决策：[ADR-007 GPTQ](../adr/007-gptq-integration.md)、
  [ADR-008 混合精度](../adr/008-mixed-precision-quantization.md)、
  [ADR-009 AWQ](../adr/009-awq-integration.md)、
  [ADR-010 SmoothQuant](../adr/010-smoothquant-integration.md)
- FAQ：[量化相关问题](../faq.md#量化-quantization)
