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

## 在模型配置中选择稀疏方案（ModelConfig.attn_sparse）

上面的 dispatcher 只负责“按名构建掩码”。要在训练 / 推理里真正把某个稀疏方案作为
模型选项使用，可以在 `ModelConfig` 上设置 `attn_sparse`（`dict[str, Any] | None`），
`kind` + 该 builder 的参数一起即可，例如：

```python
from llm.training.core.config import ModelConfig

cfg = ModelConfig(
    vocab_size=32000, hidden_size=4096, num_layers=32, num_heads=32, max_seq_len=4096,
    attn_sparse={"kind": "streaming", "num_sink": 4, "window_size": 512},
)
```

`ModelConfig` 会在校验阶段检查 `kind` 是否为 `SUPPORTED_KINDS` 之一、且必须存在。
`ModelFactory.from_config(cfg)` 会把该配置透传给 `DecoderModel`（保存在
`model.attn_sparse`）。随后任何一次前向，只要调用方没有显式传 `attn_mask`，
`DecoderModel.forward` 就会用当前序列长度自动构造派发后的掩码：

```python
from llm.runtime.model_factory import ModelFactory

model = ModelFactory.from_config(cfg)
logits = model(input_ids)  # 同一前向自动应用 streaming 掩码
```

要点：

- `build_config_attention_mask(config, seq_len)` 生成的是**纯 pattern** 掩码
  （`causal=False`）；因果性由 decoder 自身的前向（`is_causal`）施加。这样
  全 coverage 稀疏（如 `streaming` 的 `num_sink >= seq_len`）与稠密因果完全一致
  —— 这是“稀疏是稠密的约束”这一 CPU parity 不变量。
- 显式传入的 `attn_mask` 优先级更高，会覆盖自动构造的稀疏掩码。
- CPU parity 验证在 `tests/core/attn/test_sparse_model_config.py`：
  genuinely 稀疏会改变 decoder 输出，而全 coverage 稀疏与稠密逐位一致。

持久化与往返：`attn_sparse` 是 model-defining 字段，随 `save_pretrained` 写入
HF `config.json`，并在 `from_pretrained` 时还原（走
`weight_mapping.get_config_mapping` → `hf_loader.from_pretrained`），因此稀疏模型
可以 训练 → 保存 → 重载 → 推理 全链路保留其稀疏方案，不会重载后退化为稠密。
往返 parity 验证在 `tests/compat/test_hf_loader.py::test_sparse_attention_roundtrip`
（重载后的 forward 与保存前逐位一致；全 coverage 方案同样保留稠密等价性）。

解码（KV-cache generation）：`DecoderModel.forward` 自动构造掩码时按**有效 key
历史长度**（`key_len = cache 长度 + 当前 token`）生成 `[Sq, Sk]` 矩形掩码，而不是
按 1-token 输入生成退化方形掩码——否则 sink + window 永远无法约束已缓存的过去
key，长程流式生成的稀疏方案形同虚设（RIL TASK-245）。eager 生成循环因此能
端到端工作：稀疏 decode 会改变输出，而全 coverage 稀疏 decode 与稠密 decode
一致（CPU parity 验证在 `tests/core/attn/test_sparse_model_config.py`。

服务端（batched/paged serving）：`ContinuousBatchingEngine` 总是为每 batch 自己
构造 `run_attn_mask`，因此 decoder 的自动稀疏掩码不会触发。当被服务的模型带有
`attn_sparse` 时，engine 会把该方案的 sink+window pattern 按**绝对位置**折进
`run_attn_mask`（与因果掩码 OR 合并），从而在 batched/paged 服务路径真正约束
key（RIL TASK-246）。CPU parity 验证在
`tests/serving/test_engine.py::test_engine_folds_sparse_attention_into_run_attn_mask`：
同权重下稀疏 prefill 改变输出，全 coverage 稀疏与稠密逐位一致，掩码本身也会把
旧的非 sink/窗口外 key 标为 masked。

服务端性能：该稀疏 pattern 掩码随 engine 生命周期不变，因此只**构建一次并缓存**
（键控 scheme + `k_len`，scheme 变更或 resize 时失效），不再每个 decode 步重算
`[k_len, k_len]` 的布尔掩码；且缓存落在 `self.device`，避免 builder 返回 CPU 张量
与 CUDA 因果掩码 OR 时的设备不匹配（RIL TASK-247）。验证在
`tests/serving/test_engine.py::test_engine_caches_sparse_pattern_mask`。
