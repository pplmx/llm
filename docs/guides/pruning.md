# 模型剪枝 (Weight Pruning)

权重剪枝把预训练模型里每个 `nn.Linear` 权重矩阵的一部分元素置零（配一张二值
`weight_mask`），其余权重保持不变。剪枝后模型**结构与 dtype 完全一致**，推理时
计算 `x @ (W * M) + b`，因此可以在任意设备上直接运行。它是 ROADMAP 13.4「模型剪枝」
的实现，与已完成的 QAT / 知识蒸馏互补。

## 策略

`llm-prune` / `PruningConfig` 支持两种策略：

- `magnitude`（默认）：保留 `|W|` 最大的那部分元素——直觉是大权值承载更多信号。
- `random`：按 ratio 随机保留，可加 `--seed` 复现。

## 用法

```bash
llm-prune \
    --model model.pt \        # torch.save 的 DecoderModel blob
    --output pruned.pt \
    --ratio 0.5 \             # 每个 Linear 置零比例 (0, 1)
    --method magnitude        # magnitude | random
    [--target-modules fc1,fc2]  # 可选，只剪这些模块
    [--seed 7]                # 可选，仅 random 复现用
```

实现见 `llm/quantization/prune.py`（`PrunedLinear` + `prune_model` +
`compute_sparsity`），Python API：

```python
from llm.quantization.prune import PruningConfig, prune_model, compute_sparsity

sparsity = prune_model(model, PruningConfig(ratio=0.5, method="magnitude"))
print(compute_sparsity(model))  # 报告实际整体稀疏度
```

## 推理 / 服务

`llm-prune` 输出的是裸 `torch.save` 模块 blob，与 `llm-quantize` 产物走同一套
安全加载路径：`register_framework_safe_globals` 已覆盖 `llm.quantization`，
`llm-serve` / serving loader 在 `weights_only=True` 下即可加载剪枝后的模型并正常
前向。`llm/quantization/prune.py` 里 `PrunedLinear` 的 mask 会随权重一起持久化。

## 测试

`tests/quantization/test_prune.py` 与 `tests/cli/test_prune_cli.py` 覆盖：config
校验、magnitude/random 掩码正确性、`target_modules`、复现性、无 Linear 报错、前向
有限且形状保持，以及一个真实 e2e——在固定循环语料上过拟合小 decoder，轻度剪枝
（ratio 0.1）精度基本保持、重度剪枝（0.95）精度崩坏，并用 serving loader 完成
剪枝产物加载 + 前向验证。
