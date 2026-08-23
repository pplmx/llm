# 统一稀疏注意力 API（dispatcher）

阶段十五 15.1/15.2 里落地了四种稀疏 / 流式注意力掩码工具。为了让调用方用一个入口就能
按名选择，`llm/core/attn/sparse.py` 提供一个统一的派发函数
`build_sparse_attention_mask(kind, seq_len, *, causal=True, **kwargs)`。

## 支持的 kind

| kind           | 底层函数                                | 关键参数                                                                        |
| -------------- | --------------------------------------- | ------------------------------------------------------------------------------- |
| `block_sparse` | `block_sparse.build_block_sparse_mask`  | `block_size`, `window_blocks`, `global_blocks`, `random_blocks`, `seed`         |
| `streaming`    | `streaming_llm.build_streamingllm_mask` | `num_sink`, `window_size`                                                       |
| `longformer`   | `longformer.build_longformer_mask`      | `window_size`, `dilation`, `num_global`                                         |
| `bigbird`      | `big_bird.build_bigbird_mask`           | `block_size`, `num_global_blocks`, `window_blocks`, `num_random_blocks`, `seed` |

未知 `kind` 会以 `ValueError`（列出所有支持项）失败；参数校验由各底层 builder 负责。

## 用法

```python
from llm.core.attn.sparse import build_sparse_attention_mask, mask_to_additive

mask = build_sparse_attention_mask(
    "bigbird", seq_len=64, block_size=8,
    num_global_blocks=2, window_blocks=1, num_random_blocks=3,
    causal=True,
)
additive = mask_to_additive(mask)  # 0 / -inf，喂给任意 attn backend
```

## 测试

`tests/core/attn/test_sparse_dispatcher.py` 覆盖：`SUPPORTED_KINDS` 注册表；未知 kind
报错；每个 kind 派发结果 === 直接调用底层 builder；builder 校验透传；以及每个 kind
经真实 `sdpa` backend 与显式掩码逐位一致。
