# Soft MoE（研究切片）

阶段 15.4 的「实现 Soft MoE」。Soft MoE（Puigcerver et al., *From Sparse to Soft
Mixtures of Experts*, 2023）用确定、可微的 **slot** 机制取代离散 top-k 路由：

每个专家（`E` 个）拥有若干 slot（`S` 个）；每个 slot 对全部 token 做一次 softmax
归一化的软分配 `D[e,s,t]`（沿 token 维），于是：

```text
slot 输入   = Σ_t D[e,s,t] · x_t          # 任意 token 都能进任意 slot，(不丢 token、无 hard argmax)
slot 输出   = expert_e(slot 输入)
y_t        += Σ_{e,s} D[e,s,t] · slot输出[e,s]   # 按同一 D 遣回
```

这里把 dispatch 与 combine 权重**绑定为同一个** `D`（文档中的简化；论文用可学习的
slot-logits 投影，可作后续）。

关键性质（相对离散路由）：**结构性 slot 均衡**——每个 slot 的 `Σ_t D[e,s,t] == 1`，
不靠 aux loss 就能避免 slot 饿死/过载；且全链路 smooth、可微，规避 hard top-k 的负载
均衡不连续。

## 用法

```python
from llm.core.moe.soft_moe import dispatch_weights, soft_moe_output

E, S, T, d = 4, 2, 8, 8
slot_logits = torch.randn(E * S, T)  # 每 (expert, slot) 一行、每 token 一列
x = torch.randn(T, d)

D = dispatch_weights(slot_logits, E, S)  # [E, S, T]，每 slot 沿 token softmax
y = soft_moe_output(x, lambda s: s, slot_logits, E, S)  # [T, d]
```

独立研究切片；把它作为 `MoeLayer` 的一种路由模式接入（可学习 slot-logits 投影 +
sparse/soft 切换）列为后续。

## CPU parity 不变量（见 `tests/core/moe/test_soft_moe.py`）

- 每 slot 的 dispatch 权重 `Σ_t D == 1`（结构性均衡）；
- 输出 === 逐 slot 加权和参考；
- 全链路可微（x 与 slot_logits 的 backward 梯度有限）；
- 极端极限下（单 token 分数压倒），slot 输入塌缩到该 token、输出收敛为其 expert 值。
