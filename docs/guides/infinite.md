# Infinite Attention（压缩记忆研究切片）

阶段十五 15.1 的收尾项（「研究 Infinite Attention」）。Infini-Attention（Munkhdalai
et al., *Leave No Context Behind*, 2024）用一块**压缩记忆**给因果点积注意力补上"无限"
上下文：一个 `[d_k, d_v]` 的键值矩阵 `M` 加一个 `[d_k]` 归一化向量 `z`，通过非负特征映射
`φ(x)=elu(x,alpha=1)+1` 把每个过去的 (key, value) 线性累积进状态。

查询 `q` 的记忆检索头

```text
A_mem(q) = (φ(q)ᵀ M) / (φ(q)ᵀ z)
```

恰好是对已累积前缀的**线性注意力加权平均**；它与本地（因果）点积注意力用标量门控
`β` 融合：

```text
o = sigmoid(β) · A_dot + (1 - sigmoid(β)) · A_mem
```

## 用法

```python
from llm.core.attn.infinite import InfiniMemory, infinite_attention

q, k, v = torch.randn(6, 8), torch.randn(6, 8), torch.randn(6, 8)
out, mem = infinite_attention(q, k, v, beta=0.0, causal=True)
# mem.M / mem.z 累积了整段上下文，可与下一段接力
```

也可直接操作记忆：`mem.update(k, v)` 压缩一段；`mem.retrieve(q)` 取上下文向量；
`mem.reset()` 清零。

## CPU parity 不变量

- 空记忆检索为 `0`；门控推向点积侧（`β→+∞`）时输出 === 当前段的稠密 causal/full 注意力
  —— Infini-Attention 是稠密注意力的"超集"；
- 记忆检索 === 显式线性注意力参考 `Σφ(q·k_i)v_i / Σφ(q·k_i)`；
- 记忆状态是固定 `[d_k, d_v]`，与已写 token 数无关（无限上下文的关键性质）；
- 全链路可微（backward 得出有限梯度）。

验证在 `tests/core/attn/test_infinite.py`。这是**独立研究切片**；把它接入具体
`attn_impl`/模型前向（记忆让上下文实际无界，涉及独立集成设计）列为后续。
