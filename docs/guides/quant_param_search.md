# 量化参数搜索（研究切片）

阶段 13.2「优化量化参数搜索」。定点/静态量化有若干旋钮——位宽（4/8）与 scale
粒度（per-tensor / per-channel）——而"该用哪种"取决于张量的权重/激活分布。仓库的
`calibration.py` 只给 absmax 的 scale，**不做**配置空间的搜索。

本切片回答：**对给定张量，哪种 `(bits, granularity)` 重建误差最小？** 它对每个候选用
真实的 `FakeQuantize` 做一次 fake 往返，按归一化 MSE `||x - dequant(x)||² / ||x||²`
打分，返回最优配置——这是 QAT / 静态舍入里"参数搜索"的成分，CPU 可验证。

## 用法

```python
from llm.quantization.param_search import search_quant_params, reconstruction_errors

w = torch.randn(8, 64).abs() * torch.tensor([0.1, 1.0, 5.0, 0.5, 2.0, 10.0, 0.3, 1.0])[:, None]
best, best_mse, all_errs = search_quant_params(w)          # e.g. (8, 'per_channel')
errs = reconstruction_errors(w)                            # 全部候选的归一化 MSE
```

`param_candidates()` 给出 `(bits, granularity)` 候选空间（默认 4/8 × per_tensor/
per_channel）。

## CPU parity 不变量（见 `tests/quantization/test_param_search.py`）

- bit 越多误差**不增**（8-bit 相对 MSE ≤ 4-bit）；
- per-channel（更细 scale）误差 **≤** per-tensor；
- 返回的 `best` === 全部候选里的 argmin；
- 报告的误差与对同一张量重新跑 `FakeQuantize` 逐位一致。
