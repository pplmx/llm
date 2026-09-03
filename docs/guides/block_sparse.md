# Block Sparse Attention（掩码切片）

BigBird / Longformer 式 **Block Sparse Attention**：让一个 query 只关注选定的 **key 块**
（若干块组成的组），而不是所有 key——从而把注意力的计算/显存从 O(S²) 压到近线性。
ROADMAP 阶段十五 15.2 的落地：一个 **CPU 可验证的块稀疏掩码切片**，构建在既有
additive/bool attention-mask 机制之上（Sliding Window 已实现，本切片在其基础上叠加
global / random 块）。代码在 `llm/core/attn/block_sparse.py`。

## 概念

`build_block_sparse_mask(seq_len, *, block_size, window_blocks, global_blocks, random_blocks, seed, causal)`
返回一个 `[S_q, S_k]` 的布尔 mask（`True`=可 attend）。一个 query 可以 attend：

- **local window**：与 query 所在块相邻 `window_blocks` 个块内的 key（含自身块/对角线）；
- **global**：前 `global_blocks` 个 key 块（全局 token 被所有位置 attend，也 attend 所有位置）；
- **random**：固定、按种子确定的 `random_blocks` 个 key 块（BigBird 风格，确定可复现）；
- `causal=True` 时额外禁止关注未来 key。

配套：

- `mask_to_additive(mask)`：`True→0`、`False→-inf` 的加法 bias，直接喂给任意 attn backend
  （softmax 后被屏蔽位置权重为 0）。
- `coverage_fraction(mask)`：稀疏 pattern 实际放开的比例。

## CPU 关键不变量（parity）

当 window + global 一起覆盖所有 key 块时，块稀疏 mask **与稠密 causal（或全连接）mask
逐位一致**——块稀疏是对稠密注意力的**约束**，不是不同的计算。这保证了"稀疏只有在真正
屏蔽某些块时才改变结果"，避免与稠密行为意外偏离。

## 用法

```python
from llm.core.attn.block_sparse import build_block_sparse_mask, mask_to_additive

mask = build_block_sparse_mask(seq_len=64, block_size=8, window_blocks=2, global_blocks=1, random_blocks=1)
additive = mask_to_additive(mask)  # attn_bias: 0 / -inf
```

## 测试

`tests/core/attn/test_block_sparse.py` 覆盖：参数校验；global/window 全覆盖 === 稠密
causal 的 parity；非 causal 全覆盖全 1；global 恒被 attend；稀疏块 softmax 权重为 0；
causal 屏蔽未来；random 块种子确定；`mask_to_additive` 校验。

**端到端验证**：`tests/core/attn/test_sparse_attention_integration.py` 证明这些掩码确实
作用于**真实注意力计算**——喂进仓库的 `sdpa` backend 与显式手动掩码的注意力逐位一致，
且走 `MultiHeadAttention.forward` 时稀疏会改变输出、全覆盖与稠密逐位一致。
