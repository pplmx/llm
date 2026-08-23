# GRPO (Group Relative Policy Optimization)

GRPO（Shao et al. 2024）是一种分组相对的对齐策略优化：对同一 prompt 采样 `G` 条
response，把每条 response 的奖励在该组内做 **z-score 归一化**（组内相对优势
`A_i = (r_i - mean_r) / (std_r + eps)`），再用裁剪的重要性加权策略损失 + 可选的
KL-to-reference 惩罚去优化策略。它是 ROADMAP 阶段十一「其他对齐技术」的一部分，
与既有 SFT / DPO / PPO / reward 组成对齐序列。

## 组件（`llm/training/rlhf/grpo.py`）

- `group_advantages(rewards)`：按组（每行一个 prompt、`G` 个 response）计算 z-score 优势。
- `GRPOLoss(clip_eps, kl_beta)`：每个 token
  `ratio = exp(log_p_policy - log_p_old)`，取
  `min(ratio, clip(ratio)) * A` 作为 surrogate，减去 K3 KL 惩罚
  `beta * mean(exp(ref-pol) - (ref-pol) - 1)`。

## 用法

`--task grpo`（`GRPOTask` + `GRPODataModule`，标准训练循环）：

```bash
uv run llm-train --task grpo --config-path your_grpo.yaml
```

训练配置（`TrainingConfig`）相关字段：

```yaml
training:
  batch_size: 8          # 必须为 grpo_group_size 的倍数
  grpo_group_size: 4     # 每组 response 数
  grpo_clip_eps: 0.2     # ratio 裁剪范围 (1-eps, 1+eps)
  grpo_kl_beta: 0.01     # KL-to-reference 权重 (0 关闭)
```

`GRPODataModule` 是合成、CPU 可验证的：每组第一个 response 是目标（reward=1），其余
reward=0。batch 提供 `input_ids / labels / rewards`（顺序稳定、单批），任务在第一步
快照初始策略的 log-probs 作为 GRPO 的 `old`，并保持一个冻结的 reference 用于 KL。

## 测试

`tests/training/test_grpo.py` 覆盖：组内 z-score 归一化正确性、`GRPOLoss` 参数校验与
裁剪/边界行为、`--task grpo` 注册、DataModule 批量/奖励结构、非对齐 batch 报错，以及
一个 CPU e2e——GRPO 训练让"每组成员奖励最高的 response 被策略偏好"的命中率从 0 提升到 ~1。
