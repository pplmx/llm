# Expert Choice Routing（研究切片）

阶段 15.4 的第一项（「研究 Expert Choice Routing」）。仓库已有的
`MoeLayer`（`core/moe/moe.py`）是**token-choice**：每个 token 用 gate 选它
`top_k` 个专家。Expert Choice Routing（Zhou et al., *Mixture-of-Experts with
Expert Choice Routing*, 2022）反过来——**每个专家**从整批里选它 `top_k` 个 token。

它买到的关键性质是**结构性负载均衡**：每个专家恰好拿到 `k` 个 token，不会饿死
(dead) 或过载 (overloaded)，也不需要丢 token。一个 token 可以被恰好 0 个到
任意多个专家选中。

## 用法

```python
from llm.core.moe.expert_choice import expert_choice_assignment, expert_choice_output

gate = torch.randn(8, 4)  # [num_tokens, num_experts]
tokens, scores = expert_choice_assignment(gate, k=3)  # [E,k] 每专家选 top-k 个 token

out = expert_choice_output(x, gate, lambda sel: sel * 2.0, k=3)  # [T, d]
```

`expert_choice_weights(scores)` 对每个专家的 `k` 个选中 token 做 softmax 归一化
（`sum_k == 1`）；`expert_choice_output` 按 `out[t] += w[e,t]·E_e(x[t])` 把选中
专家的输出 scatter-add 回各 token。

这是一个独立的研究切片；把它作为 `MoeLayer.forward` 的另一种 routing 模式接入
（以及 sparse dispatch / 负载均衡 aux loss）是后续。

## CPU parity 不变量（见 `tests/core/test_moe.py::TestExpertChoiceRouting`）

- 每个专家的选中 token === 其 gate 分数的 `top_k`；
- 每个专家恰好 `k` 个 token（结构性负载均衡）；
- 组合输出 === 对选中 (expert, token) 对的加权和参考；
- 全链路可微（backward 得出有限梯度）。
