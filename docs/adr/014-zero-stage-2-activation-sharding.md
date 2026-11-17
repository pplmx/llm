# 014. ZeRO Stage-2: activation/gradient sharding on top of Stage-1

Date: 2026-11-18

## Status

Proposed

## Context

ROADMAP 15.5 还剩「集成 DeepSpeed ZeRO (Stage 2/3)」一项未勾。要决定怎么落地它，
先得看清现有 Stage-1 到底做到了什么、以及 Stage-2 相对它多解决什么问题。

### 现有实现（事实）

`src/llm/training/distributed/zero.py` 是一个**零依赖的原生实现**，不是 DeepSpeed
封装：`ZeroOptimizer` 继承 `torch.optim.Optimizer`，按扁平参数索引 round-robin
划分归属（`_owner_of(index, world_size)`），每个 rank 只为自己拥有的参数构建内层
optimizer，因此 Adam 的一阶/二阶矩被分片到约 `1/world_size`。

梯度同步在 `_sync()` 之外：Stage-1 的假设（与上游 DeepSpeed 相同）是**梯度在
`step` 之前已经由 DDP 平均过**。`ZeroOptimizer._sync()` 只做参数侧的收尾——对每个
参数 `all_gather` 出 `world_size` 份副本，取 owner 那一份 `copy_` 回去，让所有
rank 收敛到相同权重。

接入点集中在 `TrainingEngine`：

- `engine.py:348` —— `is_training_zero = parallel_strategy == "zero" and world_size > 1`
- `engine.py:478` —— 按 owned 子集调用 `task.build_optimizer_for(owned)` 构建
  `ZeroOptimizer`（保留任务自己的超参/分组）
- `engine.py:403` —— 拒绝非标准循环任务（`custom-loop tasks` 不走这里）
- `engine.py:548` —— 拒绝 fp16 AMP（`GradScaler` 的 `unscale_`/per-param-grad 需要
  内层 optimizer 内部状态，分片包装未暴露）
- `engine.py:1228` —— checkpoint 保存时把 ZeRO 并入「每 rank 都需要进入集合」的那
  一组，因为优化器状态是分片的，payload 必须集合式 gather

`ZeroOptimizer.collective_state_dict()` 用 `all_gather_object` 收集每 rank 的分片，
`ZERO_RANK_STATES_KEY = "__zero_rank_states__"` 是它们在 checkpoint 里的键。

### Stage-2 多解决什么

Stage-1 分片的是**优化器状态**。每个 rank 仍持有：

1. 完整的一份参数（`self._params` 是全部参数的引用）
2. 完整的一份梯度（DDP 平均后，每个 rank 都有一份全量梯度）
3. 自己的激活（这部分 Stage-2 不改）

Stage-2 额外把**梯度**也分片：每个 rank 只保留自己拥有参数的梯度，其余在
`reduce_scatter` 里直接归约进 owner，不落全量。显存收益是每 rank 少一份完整梯度
（`num_params * dtype` 字节），对大模型是实打实的一块。

### 约束（不可绕过的仓库事实）

- `docs/adr/003-use-kv-cache.md` 等既有 ADR 都强调「纯 CPU 可验证」这个基调；本仓
  的分布式测试全部能在 gloo/CPU 上跑（`tests/training/distributed/`）。
- `AGENTS.md` §2.3：跨界前 MUST 先落 ADR —— 本文件就是这一步。
- 本机 `flash_attn` 装不了，但 ZeRO 与 attention 后端无关，不受影响。
- 现有 ZeRO 测试只覆盖 Stage-1 语义（参数 in-place 更新 + all_gather 后一致性）。

## Decision

**分两步落地，先 Stage-2，且不引入 DeepSpeed 依赖。**

### 1. 保持原生实现，不接 DeepSpeed

ROADMAP 写的是「集成 DeepSpeed ZeRO」，但把 DeepSpeed 拉进来会带来：一个新的大依赖
（含 CUDA 扩展编译）、它自己的 optimizer/checkpoint 格式、和现有 `ZeroOptimizer`
的 `state_dict` 契约冲突。而 Stage-2 的核心机制（`reduce_scatter` + 本地梯度切片）
用 `torch.distributed` 就能实现，且能在 gloo/CPU 上验证——这与本仓既有的分布式测试
能力一致。

因此：**实现 `ZeroStage2Optimizer`，语义对齐 DeepSpeed 的 Stage-2，但作为本仓原生
组件；是否真的依赖 DeepSpeed 留到 Stage-3（需要参数分片 + 前向/反向重算，复杂度
高一个量级）再单独决策。** ROADMAP 的措辞应随之细化。

### 2. Stage-2 的具体机制

梯度分片走 `reduce_scatter_tensor`（gloo 在 torch 2.x 支持；CPU 上退化为
allreduce+slice 的语义等价实现，仅用于测试）。

每个 step：

1. **反向结束**：得到每个 rank 自己的局部梯度（它那份 microbatch 的贡献）。
2. **归约**：`reduce_scatter_tensor` 把「全量梯度和」分片——rank `r` 拿到自己
   owned 参数那一段的**全局平均**梯度。
3. **更新**：owner 用自己的分片梯度做内层 `optimizer.step()`。
4. **广播参数**：复用 Stage-1 的 `_sync()`（all_gather + owner 切片），保证每个
   rank 的权重一致。

关键点：第 2 步之后，非 owner 的参数**没有梯度**，必须把它当作「本 rank 不参与更新」
而不是「梯度为零」——否则后续任何假设「梯度可读」的代码（梯度裁剪、逐参数监控、
梯度范数上报）都会静默地拿到全零。

### 3. 与现有代码的接缝

- **配置**：`DistributedConfig` 增 `zero_stage: int = 1`，取值 `1 | 2`；`parallel_strategy="zero"`
  保持不变。不新增 strategy 名字，避免与 tp/pp/3d 的组合矩阵爆炸。
- **梯度裁剪**：Stage-2 下全局梯度范数必须在 `reduce_scatter` **之前**算（那时每个
  rank 还持有全量局部梯度），用 `all_reduce` 求平方和再开方。这正是
  `engine.py` 现在 `clip_grad_norm_tp` / `clip_grad_norm_` 的调用位置附近要处理的分支。
- **检查点**：沿用 `collective_state_dict()` 与 `ZERO_RANK_STATES_KEY`，Stage-2 的
  优化器状态形状与 Stage-1 一致（仍然是 owned 子集），无需新格式。
- **AMP**：fp16 继续拒绝（理由不变）；bf16/fp32 可用。

### 4. 明确不做的事

- 不做 Stage-3（参数分片 + 前向/反向重算）。它要求模型在 forward 前按层 gather
  参数、backward 后立即丢弃，会侵入 `DecoderModel.forward` 与 PP/TP 的现有实现，
  属于另一个量级的改动，需要独立 ADR。
- 不把 `parallel_strategy="zero"` 与 tp/pp/3d 组合。Stage-2 与 3D 并行的交互
  （梯度分片跨哪些组）需要先设计 rank 网格语义，本 ADR 只覆盖纯 DP + ZeRO。

## Consequences

### Positive

- 每个 rank 省下一整份梯度的显存（`num_params * dtype` 字节），这是 Stage-1 拿不到的。
- 保持零依赖：不需要编译 DeepSpeed 的 CUDA 扩展，`uv sync` 不增加负担。
- 可在 gloo/CPU 上验证语义正确性，与既有 `tests/training/distributed/` 的基调一致。
- 复用现有接缝：`ZeroOptimizer` 的参数归属、`_sync()`、checkpoint 契约都不变，
  改动集中在梯度归约这一步。

### Negative

- **与 DeepSpeed 不是二进制兼容**。ROADMAP 原话「集成 DeepSpeed ZeRO」若被理解成
  「能加载 DeepSpeed 的 checkpoint」则不满足；本决策交付的是语义对齐的原生实现。
  这一点必须在 ROADMAP 里写清楚，否则读者会误判。
- Stage-2 的梯度分片让「梯度」不再是每个 rank 都能读到的全量数据。仓库里任何
  读取 `.grad` 做诊断/上报的代码都需要检查一遍，包括 `performance_monitor` 的
  梯度范数路径和逐参数监控。
- CPU 上没有真正的 `reduce_scatter_tensor` 走法时，测试用的等价实现要单独维护，
  存在「测试路径和生产路径不同」的风险。缓解方式：在至少一次真实多卡 GPU 运行上
  验证，而不只靠 CPU 等价实现。

### Neutral

- `DistributedConfig` 多一个 `zero_stage` 字段；已有配置默认 `1`，行为不变。
- `ZeroOptimizer` 需要一个 Stage-2 变体（新增类，不改 Stage-1 的），二者共享参数
  归属与 `_sync`。

## References

- ROADMAP §15.5 高级分布式训练 — 未勾的「集成 DeepSpeed ZeRO (Stage 2/3)」
  （ROADMAP.md 不在 mkdocs 文档树内，按本目录其他 ADR 的惯例以纯文本引用 §编号）
- `src/llm/training/distributed/zero.py` — 现有 Stage-1 实现
- `src/llm/training/core/engine.py:348,403,478,548,1228` — 接入点
- `tests/training/distributed/test_zero_optimizer.py` — Stage-1 语义测试
- Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter
  Models* (SC 2020) — Stage 1/2/3 的划分依据
- [ADR-003: Use Pre-Allocated KV Cache](003-use-kv-cache.md) — 本仓「纯 CPU 可验证」
  基调的来由
