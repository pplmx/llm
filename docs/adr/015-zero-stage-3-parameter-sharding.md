# 015. ZeRO Stage-3: parameter sharding with on-the-fly gather

Date: 2026-11-18

## Status

Proposed

## Context

ROADMAP 15.5 的「集成 DeepSpeed ZeRO (Stage 2/3)」目前的状态：Stage-1 与 Stage-2 都已
落地并验证（[ADR-014](014-zero-stage-2-activation-sharding.md)），Stage-3 未做。本
ADR 决定怎么落地它。

### Stage-3 相对 Stage-2 多解决什么

Stage-2 之后每 rank 仍持有：

1. **完整的一份参数**（`self._params` 是全部参数的引用）
2. 自己的激活
3. 自己那份优化器状态（1/world_size）
4. 自己那份梯度（1/world_size）

Stage-3 额外把**参数**分片：每 rank 只长期持有 1/world_size 的参数，训练时按需
gather、用完即弃。显存收益是每 rank 少 `(world_size-1)/world_size × num_params × dtype`
字节——对大模型这是最大的一块。

### 现有代码的约束（事实）

**`DecoderModel.forward` 以整模型为单位，没有按层的参数钩子**。`src/llm/models/decoder.py`
的 `forward` 里已经有逐层循环：

```python
for i, block in enumerate(self.transformer_blocks):
    ...
    block_outputs = block(hidden_states, attn_mask=attn_mask, is_causal=None, ...)
```

但参数在循环**之外**由 `model.parameters()` 一次性给出——`TrainingEngine` 里三处都这么用
（`engine.py:491` 构建优化器、`engine.py:920` 遍历梯度、`engine.py:1006` 裁剪）。
Stage-3 要在 forward 前把某一层的参数 gather 回显存、backward 后立刻丢弃，就必须在这个
循环上做文章，而不是在模型外面包一层。

**梯度检查点已经存在**且与 Stage-3 冲突面重叠：`decoder.py:268` 在
`_gradient_checkpointing and self.training` 时用 `torch.utils.checkpoint.checkpoint(block, ...)`
重算前向。两个机制都靠「重算/重放前向」省显存，组合起来的重算次数是乘法的。

**PP / TP / 3D 也改参数布局**。`partition_decoder_model` 按 `transformer_blocks` 切 stage
（PP），`apply_tensor_parallel` 把每个 linear 切片（TP）。Stage-3 若与它们组合，需要
同时定义「参数属于哪个 stage、哪个 TP shard、哪个 ZeRO owner」——三维归属问题。

**当前 ZeRO 的策略互斥**：`parallel_strategy` 是单值枚举（`ddp|fsdp|tp|pp|3d|zero`），
ZeRO 不能与 tp/pp/3d 同时启用。Stage-3 保持这个互斥。

### 为什么不能照搬 DeepSpeed

DeepSpeed 的 Stage-3 依赖它自己的 engine/module 包装层与 partition 参数容器，并深度
调用其通信调度器。本仓是零依赖原生实现（ADR-014 已定调），引入 DeepSpeed 会带来 CUDA
扩展编译、自己的 checkpoint 格式、与现有 `state_dict` 契约冲突。因此 Stage-3 继续原生。

## Decision

**分两步落地，且不在这一版做完整 Stage-3。**

### 1. 先做「参数分片 + 逐层 gather」，不做跨层的预取流水

具体机制：

1. **长期存放**：每 rank 只在 GPU 上保留自己 owned 的参数 shard；其余参数放在
   **pinned CPU 内存**（`pin_memory=True`），不占显存。
2. **前向**：进入 `for i, block in enumerate(self.transformer_blocks)` 之前，把该层
   的参数 gather 到显存（`all_gather` 或从 pinned 内存 H2D 拷贝，取决于该层是否被分片）；
   前向结束、反向结束（该层梯度已算出）后立刻释放。
3. **反向**：需要重放该层前向（因为参数已被丢弃）。这就是与**梯度检查点同源**的机制
   ——两者可以共用同一次重算：`use_reentrant=False` 的 `checkpoint` 已经保存了输入，
   重算时只要保证参数在该层 gather 窗口内可见。

关键设计点：**重算策略复用现有的 `torch.utils.checkpoint` 路径**，而不是自己写一套
autograd 钩子。这要求 `decoder.py` 的循环里有一个「进入该层前」和「离开该层后」的回调点。

### 2. 明确拒绝的组合（这一版不做）

- **Stage-3 + PP/TP/3D**：三维参数归属需要先设计；`parallel_strategy` 保持单值互斥。
- **Stage-3 + fp16 AMP**：与 Stage-1/2 同样的理由（`GradScaler` 需要看到梯度）。
- **Stage-3 + 非标准循环任务**：同 Stage-1/2。

### 3. 与现有代码的接缝

- **配置**：`DistributedConfig.zero_stage` 扩到 `1 | 2 | 3`（当前 `le=2`）。
- **模型侧**：给 `DecoderModel.forward` 的逐层循环加一层可选的
  「层生命周期钩子」。**这是本 ADR 里最需要谨慎的一处**——它把分布式概念漏进模型代码。
  替代方案是在 `TrainingEngine` 里用一个包装模块替换 `transformer_blocks` 的迭代，
  但那样要复制整个 forward，两份实现会漂移。倾向钩子，且钩子默认 `None`（零开销）。
- **检查点**：Stage-3 的参数在多数时候不在 GPU 上，`state_dict` 需要按 owner 收集
  全部分片（复用 `collective_state_dict` 的模式）。
- **梯度裁剪**：与 Stage-2 一致，必须先算全局范数再归约；Stage-3 下梯度同样只有 owned
  部分常驻。

## Consequences

### Positive

- 每 rank 省下约 `(world_size-1)/world_size` 的参数显存，这是 Stage-2 之后剩下的最大一块。
- 复用现有 `torch.utils.checkpoint` 重算路径，而不是新写一套 autograd 钩子。
- 保持零依赖与 CPU/gloo 可验证（小模型上 gather 语义与显存行为都能测）。

### Negative

- **侵入模型代码**。给 `DecoderModel.forward` 加层生命周期钩子，等于让模型知道分布式
  的存在。这与本仓「模型不感知并行」的现状相冲突，是本 ADR 最需要评审的一处。
- **重算开销**。Stage-3 的前向要跑两遍（一遍算激活、一遍算梯度），FLOPs 增加约 33%。
  与梯度检查点叠加时还会再乘一次，因此两者同时开启需要实测确认没有退化到不可用。
- **实现复杂度显著高于 Stage-2**。Stage-2 是「多一个 collective + 丢梯度」，Stage-3 是
  「改变参数的生命周期」，调试难度不在一个量级。

### Neutral

- `zero_stage` 多一个取值；默认仍为 1。
- 与 `gradient_checkpointing` 的交互需要单独设计，可能最终是「Stage-3 隐含开启重算，
  于是忽略该开关」。

## References

- ROADMAP §15.5 高级分布式训练 — 「集成 DeepSpeed ZeRO (Stage 2/3)」的剩余部分
- [ADR-014: ZeRO Stage-2](014-zero-stage-2-activation-sharding.md) — Stage-1/2 的
  实现与两处实测修正
- `src/llm/models/decoder.py:264-292` — Stage-3 必须介入的逐层循环
- `src/llm/models/decoder.py:268` — 现有梯度检查点路径（Stage-3 的重算可复用它）
- `src/llm/training/core/engine.py:491,920,1006` — 三处 `model.parameters()` 全量遍历
- `src/llm/training/distributed/zero.py` — Stage-1/2 实现
- Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter
  Models* (SC 2020) — Stage-3 的原始设计（参数分片 + 重算）
