# Auxiliary Load-Balancing Loss（研究切片）

阶段 15.4 的「优化 Expert Load Balancing」。仓库现有的 token-choice `MoeLayer`
（`core/moe/moe.py`）没有任何均衡机制：gate 可能塌缩，让少数专家霸占大部分 token、
其余专家变 dead，损害训练稳定与容量利用。标准缓解是 **Switch Transformer /
ST-MoE 辅助负载均衡损失**：

```text
p    = softmax(gate_logits, dim=-1)   # 每 token 的专家概率 [T, E]
idx  = argmax(p, dim=-1)              # 每 token 的 top-1 路由专家 [T]
f_i  = (1/T) · count(idx == i)        # 路由到专家 i 的 token 占比 [E]
P_i  = (1/T) · Σ_t p[t, i]            # 专家 i 的平均派发概率 [E]
L_aux = aux_weight · E · Σ_i f_i · P_i
```

`f_i·P_i` 只有在专家 i **既收到很多 token、又被 gate 强烈偏爱**时才大；最小化它就
鼓励 gate 把负载摊开。以小权重 `aux_weight` 加进总损失即可。

## 用法

```python
from llm.core.moe.load_balance import load_balancing_loss, routing_fractions, mean_dispatch_probability

gate = torch.randn(64, 4)               # [T, E] gate logits
fracs = routing_fractions(gate, 4)      # f: 每专家 token 占比
probs = mean_dispatch_probability(gate, 4)  # P: 每专家平均派发概率
loss = load_balancing_loss(gate, 4, aux_weight=0.01)  # 标量，可加进总损失
```

这是独立研究切片；把它接进 `MoeLayer`（`L_total += aux_weight·L_aux` 并回传）列为后续。

## CPU parity 不变量（见 `tests/core/moe/test_load_balance.py`）

- 接近均匀（balanced）的 gate 损失严格小于全押一个专家（imbalanced）——损失随
  不均衡单调上升；
- `f`、`P` 各自归一到 1；
- 对 `gate_logits` 可微（backward 梯度有限）；
- `aux_weight` 线性缩放（0 为 no-op）。
