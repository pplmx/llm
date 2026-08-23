# Longformer-style Dilated Sliding-Window Attention（掩码切片）

Longformer（Beltagy et al. 2020）用 **dilated sliding window** + 少量 **global token**
把注意力压到近线性：窗口内每个 query 只 attend 每隔 `dilation` 步的 key（同样的预算
能覆盖更远），而前 `num_global` 个位置被所有 token attend。

ROADMAP 阶段十五 15.2 的落地：一个建在既有 additive/bool attention-mask 机制之上的
**CPU 可验证的稀释滑窗掩码切片**（与 block-sparse / StreamingLLM 共享 `mask_to_additive` /
`coverage_fraction`）。代码在 `llm/core/attn/longformer.py`。

## 概念

`build_longformer_mask(seq_len, *, window_size, dilation=1, num_global=0, causal=True)`
返回 `[S_q, S_k]` 布尔掩码。一个 query 可 attend：

- **global**：前 `num_global` 个 key 位置（所有 token 都 attend）；
- **dilated window**：`|query - key| <= window_size` **且** `(query - key) % dilation == 0`；
- `causal=True` 额外禁止关注未来 key。

## CPU 关键不变量（parity）

`dilation=1`、`num_global=0` 时，该掩码**与普通滑窗掩码逐位一致**——dilation 是叠加在
窗口之上的约束而非不同计算；且稀释掩码是稠密窗口掩码的**严格子集**。

## 用法

```python
from llm.core.attn.longformer import build_longformer_mask
from llm.core.attn.block_sparse import mask_to_additive

mask = build_longformer_mask(seq_len=64, window_size=16, dilation=2, num_global=4)
additive = mask_to_additive(mask)  # attn_bias: 0 / -inf
```

## 测试

`tests/core/attn/test_longformer.py` 覆盖：参数校验；`dilation=1`+无 global === 普通
滑窗（parity）；global 恒被 attend；`dilation>1` 覆盖率降低且是稠密窗口子集；causal
屏蔽未来；并通过仓库 `sdpa`/`MultiHeadAttention.forward` 与显式手动掩码逐位一致。
