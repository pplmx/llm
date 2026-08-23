# StreamingLLM（attention-sink 流式掩码切片）

StreamingLLM（Xiao et al. 2023）让**滚动上下文的 LLM** 在超长流上保持稳定：始终 attend
最前面的几个位置（**attention sink**）和最**近一段 window**，而把旧的、非 sink 的中间 token
从注意力里剔除——正是这些被剔除的"旧上下文"会破坏流式注意力的稳定性。

ROADMAP 阶段十五 15.1 的落地：一个建在既有 additive/bool attention-mask 机制之上的
**CPU 可验证的 attention-sink 掩码切片**（与 Block Sparse 切片共享同一套掩码工具）。
代码在 `llm/core/attn/streaming_llm.py`。

## 概念

`build_streamingllm_mask(seq_len, *, num_sink, window_size, causal=True)`
返回 `[S_q, S_k]` 布尔掩码（`True`=可 attend）。一个 query 可 attend：

- **sink**：前 `num_sink` 个位置（attention sink，恒被 attend）；
- **window**：以 query 结尾的最近 `window_size` 个位置；
- `causal=True` 额外禁止关注未来 key（即使是未来的 sink 位置也不可见）。

配套复用 `llm/core/attn/block_sparse.mask_to_additive`（`True→0`、`False→-inf`）与
`coverage_fraction`。

## CPU 关键不变量（parity）

当 `num_sink >= seq_len`（每个位置都是 sink）或 window 覆盖全部历史时，流式掩码
**与稠密 causal 掩码逐位一致**——流式注意力是对稠密注意力的**约束**而非不同计算，
非稀疏极限下两者完全一致。

## 用法

```python
from llm.core.attn.streaming_llm import build_streamingllm_mask
from llm.core.attn.block_sparse import mask_to_additive

mask = build_streamingllm_mask(seq_len=64, num_sink=4, window_size=8)
additive = mask_to_additive(mask)  # attn_bias: 0 / -inf
```

## 测试

`tests/core/attn/test_streaming_llm.py` 覆盖：参数校验；sink/window 全覆盖 === 稠密
causal 的 parity；sink 恒被 attend；旧的非 sink 中间 token 被屏蔽且 softmax 权重为 0；
causal 屏蔽未来（即使该位置在 sink 范围内）；random 无关的可复现结构。
