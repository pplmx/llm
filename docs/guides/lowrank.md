# 低秩分解 (Low-Rank Decomposition)

低秩分解把每个预训练 `nn.Linear` 的权重矩阵 `W (out x in)` 分解成两个小矩阵的乘积
`W ≈ U @ V`（`U: out x r`、`V: r x in`），中间秩 `r` 很小，从而压缩参数量。用截断
SVD（Eckart-Young）得到该秩下的最优 F 范数近似。它是 ROADMAP 13.4「探索低秩分解」
的实现，与 QAT / 知识蒸馏 / 模型剪枝共同组成压缩系列。

分解后模型结构与 dtype 完全一致，前向等价于 `(x @ V^T) @ U^T + b`，可在任意设备
运行。`llm-decompose` 包装该流程；`LowRankLinear` 被 `weights_only` 安全白名单
（`llm.quantization`）覆盖，剪枝/低秩产物可经同一 serving loader 直接加载。

## 用法

```bash
llm-decompose \
    --model model.pt \        # torch.save 的 DecoderModel blob
    --output lowrank.pt \
    --rank 8                  # 显式秩（与 --rank-ratio 二选一）
    # 或 --rank-ratio 0.25    # 自动秩 = ratio * min(out, in)
    [--target-modules fc1,fc2]  # 可选
```

Python API（`llm/quantization/lowrank.py`）：

```python
from llm.quantization.lowrank import LowRankConfig, decompose_model, compute_compression

stats = decompose_model(model, LowRankConfig(rank=8))
print(stats["compression_ratio"])  # 报告压缩比
print(stats["relative_error"])     # 报告平均 F 范数重建误差
```

## 测试

`tests/quantization/test_lowrank.py` 与 `tests/cli/test_decompose_cli.py` 覆盖：config
校验（rank / rank-ratio 二选一）、全秩分解逐位还原、低秩形状与压缩比、`target_modules`、
前向有限且形状保持，以及真实 e2e——固定循环语料上过拟合小 decoder，全秩分解精度保持、
秩 1 崩塌精度下降，并用 serving loader 完成低秩产物加载 + 前向验证。
