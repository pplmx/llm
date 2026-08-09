---
tags:
  - 指南
  - 训练
---

# Checkpoint 完整指南

训练过程中的模型保存与恢复。

## 概述

Checkpoint 保存三类状态：

- 模型权重
- 优化器 / 调度器 / scaler 状态
- 训练进度（epoch、loss）与可选的 `extra_state`（如流式数据 cursor）

### 磁盘布局（v2 split，默认）

`CheckpointManager` 把每个 checkpoint 写成**三件套**（[ADR-006](../adr/006-checkpoint-format-unification.md)）：

```text
checkpoints/
├── latest.safetensors      # 模型权重（safetensors，可零拷贝部分加载）
├── latest.meta.json        # 训练元数据（epoch / loss / best_loss / model_config）
├── latest.extra_state.pt   # optimizer / scheduler / scaler + extra_state
├── best.safetensors        # 最低 loss 模型（save_best=true 时）
├── best.meta.json
├── best.extra_state.pt
├── epoch_1.safetensors     # 按 save_interval 落盘
├── epoch_1.meta.json
└── epoch_1.extra_state.pt
```

- 权重用 safetensors：optimizer 状态损坏不会挡住模型权重读取；`*.safetensors` 可直接用于 HF 发布、评估、从预训练 base 起步。
- 元数据是 JSON：`cat latest.meta.json | jq`。
- 旧版 v0.0.5 单文件 `.pt` 仍可加载（自动检测 + `DeprecationWarning`），可用 `llm-migrate-ckpt` 迁移。

## 训练中自动保存

`llm-train` 通过 YAML 的 `checkpoint:` 段控制（**没有 CLI 参数**）：

```yaml
checkpoint:
  checkpoint_dir: checkpoints
  resume_from_checkpoint: null   # 续训时填 stem（见下）
  save_interval: 1               # 每 N 个 epoch 落盘 epoch_N.*
  keep_last_n: 5                 # 只保留最近 N 份 epoch_N.*
  save_best: true                # 额外写 best.*
```

每次保存写 `latest.*`；`save_best=true` 时最低 loss 额外写 `best.*`；`save_interval` 按 epoch 写 `epoch_N.*` 并自动轮转。

## 恢复训练（resume）

resume 通过 YAML 设置，`resume_from_checkpoint` 填 stem（无后缀）即可：

```yaml
checkpoint:
  resume_from_checkpoint: checkpoints/epoch_2
```

```bash
# 第一次跑（生成 checkpoints/epoch_2.* 三件套）
uv run llm-train --task stream_lm --config-path configs/streaming_local_demo.yaml --epochs 2

# 续训：把 resume_from_checkpoint 指向上次的 stem，重跑同一命令
uv run llm-train --task stream_lm --config-path configs/streaming_local_demo.yaml --epochs 2
```

`CheckpointManager.load_checkpoint` 恢复 model / optimizer / scheduler / scaler / `extra_state`。
流式任务额外恢复数据 cursor（不重复读、不遗漏）。

## 手动检查 / 加载

```python
from llm.training.core.checkpoint import load_checkpoint_payload

# 自动解析 v2 split 三件套；兼容旧式单文件 .pt
ckpt = load_checkpoint_payload("checkpoints/latest")
print(ckpt["epoch"], ckpt["loss"], ckpt["extra_state"])
```

返回统一字典，键：`model_state` / `model_config` / `epoch` / `loss` / `best_loss` /
`optimizer_state` / `scheduler_state` / `scaler_state` / `extra_state` / `format_version`。

## 迁移旧版 checkpoint

```bash
# 预览将做什么
llm-migrate-ckpt checkpoints/epoch_5 --dry-run

# 转换 + round-trip 校验 + 删除旧文件
llm-migrate-ckpt checkpoints/epoch_5 --verify --in-place
```

参数与退出码见 [CLI 参考 · llm-migrate-ckpt](../reference/cli.md#llm-migrate-ckpt)。

## 服务化加载

`llm-serve` 的 `LLM_SERVING_MODEL_PATH` 直接指向 checkpoint stem：

```bash
LLM_SERVING_MODEL_PATH=checkpoints/epoch_5 \
uv run llm-serve
```

v2 split 三件套与旧式 `.pt` 都支持；详见 [推理教程](../tutorials/03-inference.md)。

## 最佳实践

1. **定期保存**：`save_interval` 按 epoch 粒度；中断后用 `latest.*` 或 `epoch_N.*` 续训。
2. **保留最优**：`save_best: true` 让 `best.*` 始终指向最低 loss。
3. **清理旧 checkpoint**：`keep_last_n` 自动轮转；手动清理时删除同 stem 的三件套，不要只删单个文件。
4. **权重部分加载**：`safetensors.torch.load_file("latest.safetensors")` 跳过 optimizer 状态，
   适合只做评估、发布、或从预训练 base 起步的微调。

## 故障排除

**Q: Checkpoint 损坏？**

- 检查三件套是否齐全（缺一即视为该 checkpoint 不存在）；`llm-migrate-ckpt --verify` 可做新旧布局 round-trip 校验。

**Q: 模型不匹配？**

- 确保 resume 的 checkpoint 与当前 config 一致（`model_config` 在 `meta.json` 里可查）。
  流式任务还会校验数据源指纹，换数据集会 loud fail。

**Q: 训练中断？**

- 在 YAML 设 `checkpoint.resume_from_checkpoint: checkpoints/latest`，重跑同一命令即可。

## 相关文档

- [预训练教程](../tutorials/01-pretraining.md)
- [推理服务](../tutorials/03-inference.md)
- [ADR-006: Checkpoint 格式统一](../adr/006-checkpoint-format-unification.md)
