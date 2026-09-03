# BigBird Attention（掩码切片）

BigBird（Zaheer et al. 2020）把注意力组合成三种 pattern：少数 **global** token、一个
**local window**、以及一组固定的 **random** key 块——把全注意力从 O(S²) 压到近线性。

ROADMAP 阶段十五 15.2 的最后一项：这个 pattern 正是此前 block-sparse 掩码工具
已经实现的（`global_blocks` + `window_blocks` + 按种子的 `random_blocks`），所以本切片
在共享机制上提供一个**显式的 BigBird API**，保持同样的语义与 CPU parity 不变量。
代码在 `llm/core/attn/big_bird.py`。

## 概念

`build_bigbird_mask(seq_len, *, block_size, num_global_blocks, window_blocks, num_random_blocks=0, seed=0, causal=True)`
返回 `[S_q, S_k]` 布尔掩码，直接委托给 `block_sparse.build_block_sparse_mask`：

- **global**：前 `num_global_blocks` 个 key 块；
- **window**：与 query 块相邻 `window_blocks` 个块；
- **random**：按种子确定的 `num_random_blocks` 个随机 key 块；
- `causal=True` 额外禁止关注未来 key。

## CPU parity

`num_global_blocks >= 总块数` 时掩码 === 稠密 causal（非 causal 则全 1）；random 块按
种子可复现，且 `num_random_blocks>0` 时覆盖率严格高于纯 global+window。

## 用法

```python
from llm.core.attn.big_bird import build_bigbird_mask
from llm.core.attn.block_sparse import mask_to_additive

mask = build_bigbird_mask(seq_len=64, block_size=8, num_global_blocks=2, window_blocks=1, num_random_blocks=3)
additive = mask_to_additive(mask)  # attn_bias: 0 / -inf
```

## 测试

`tests/core/attn/test_big_bird.py` 覆盖：委托 === block-sparse；global 全覆盖 === 稠密
causal/全 1；random 增加覆盖率且种子可复现；causal 屏蔽未来；并通过仓库 `sdpa` /
`MultiHeadAttention.forward` 与显式手动掩码逐位一致（真实 BigBird 稀疏会改变输出）。
