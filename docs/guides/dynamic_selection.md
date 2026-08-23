# Dynamic Expert Selection（研究切片）

阶段 15.4 的「探索 Dynamic Expert Selection」。标准稀疏 MoE 给每个 token 固定
`top_k` 个专家；动态选择按路由置信度**逐 token 调节专家数**：top-1 概率高的
"容易" token 只需少数专家，不确定的 token 则拿到更多容量——省算力又给难 token
更多头。

本切片实现一个两档自适应方案：

```text
p    = softmax(gate_logits)           # 每 token 专家概率 [T, E]
k_t  = min_experts  当 top-1(p) >= high_conf_threshold   # 自信->少
       否则 max_experts                                  # 不确定->满预算
```

每 token 只对 top `max_experts` 专家排一次名，保留前 `k_t` 个并对它们**重新归一化**
成合法分布。全链路 softmax、无 argmax 进权重路径，完全可微。独立研究切片；把自适应
计数接入 `MoeLayer.forward`（变容量 dispatch）列为后续。

## 用法

```python
from llm.core.moe.dynamic_selection import dynamic_expert_count, dynamic_expert_output

gate = torch.randn(8, 4)               # [T, E] gate logits
kt = dynamic_expert_count(gate, min_experts=1, max_experts=3, high_conf_threshold=0.7)  # [T]
out = dynamic_expert_output(x, gate, lambda s: s * 2.0, min_experts=1, max_experts=3, high_conf_threshold=0.7)
```

## CPU parity 不变量（见 `tests/core/moe/test_dynamic_selection.py`）

- 自信 token（top-1 >= 阈值）恰用 `min_experts`，不确定的恰用 `max_experts`，计数在
  `[min,max]` 内；
- 每 token 权重和为 1（对保留专家归一化）；
- 输出 === 显式重归一化加权和参考；
- `gate_logits` 与 `x` 的 backward 梯度有限。
