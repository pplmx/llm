# DPO vs PPO (RLHF) 对比基准

阶段十一 **11.3「对比 DPO vs RLHF 性能」** 的落地并扩展：一个 CPU 可验证的基准
harness，在**同一个合成偏好任务**上分别跑既有 **DPO**、**PPO (RLHF)** 与 **GRPO**
三种对齐方式，报告可比的指标并给出观测结论。基准代码在
`llm/training/rlhf/aligner_benchmark.py`。

## 共享偏好任务

用一个 **judge** 定义"什么是好 response"：**以目标 token 结尾** 的 response 得分 1，
否则 0。这是一个干净的、CPU 可验证的偏好信号：

- **DPO** 侧：`AIFeedbackDataModule` 产出随机(chosen, rejected) 对，其中 chosen 以目标
  token 结尾、rejected 不以目标结尾；`DPOTask` 直接在这些偏好对上做 off-policy 优化。
- **PPO** 侧：把同一个 judge 实现成规则 reward 模型 `TargetTokenReward`（作为 PPO 的
  reward model），`PPOTask` 用 on-policy rollout 从 prompt 生成 response、按是否以目标
  token 结尾给奖励。
- **GRPO** 侧：`GRPOTask` 在 `GRPODataModule` 的固定合成分组上做 group-relative 优化
  （组内第一个 response 为全零目标、reward=1），用其原生 group-reward fraction 度量
  "组内最优 response 命中目标"的比例。

**共享指标**：`preference_fraction(policy, chosen, rejected)` —— 策略对 chosen 序列的
累计 log-prob 大于 rejected 的比例。它在同一份共享 set 上同时评估两个策略，保证可比。

## 组件（`llm/training/rlhf/aligner_benchmark.py`）

- `TargetTokenReward`：规则 reward——最后一个真实 token 等于目标则 1 否则 0（`(ids, mask) -> [B]`，与 reward-model 调用契约一致）。
- `preference_fraction(model, chosen, rejected)`：chosen-logp > rejected-logp 的比例。
- `run_dpo(config, epochs)`：`AIFeedbackDataModule` + `DPOTask` 标准循环，返回偏好比例轨迹 + DPO 损失轨迹。
- `run_ppo(config, steps, prompts)`：`PPOTask`（reward 为 `TargetTokenReward`）+ 真实 `PPOTrainer` rollout，返回 mean-reward 轨迹。
- `run_grpo(config, epochs)`：`GRPODataModule` + `GRPOTask`，返回 group-reward fraction 轨迹。
- `compare_dpo_vs_ppo(config, ...)`：一起跑三者并返回 `{dpo, ppo, grpo, summary}`。

## 用法

```python
from llm.training.rlhf.aligner_benchmark import compare_dpo_vs_ppo

result = compare_dpo_vs_ppo(config, dpo_epochs=40, ppo_steps=3, grpo_epochs=40, prompts=["hello", "world"])
print(result["summary"])
```

## CPU 观测结论

在这套小规模、CPU 预算内，基准给出了一个**真实且可复现**的对比：

- **DPO 稳定收敛**：off-policy、偏好对驱动的 DPO 在几十个 epoch 内把
  `preference_fraction` 从 ~0.5 可靠地抬到 >0.9（模拟先随机被打乱、后提升）。
- **GRPO 稳定收敛**：group-relative 的 GRPO 在固定合成分组任务上把 group-reward
  fraction 从 ~0 抬到 >0.9，CPU 上可复现（与其自带 e2e 一致）。
- **PPO 波动、难收敛**：on-policy rollout + 稀疏的"以目标结尾"奖励下，PPO 在同等小预算内
  噪声大、收敛慢甚至不升。这是已知的样本效率/稳定性差异在 CPU 上的真实体现
  （DPO 免 rollout、off-policy、更稳；PPO 需要足够的在线采样强度才能把稀疏奖励放大）。

因此本切片把 DPO 侧做成强断言，PPO 侧断言 harness 端到端产出有限、良构的 reward 轨迹
（不假装它在一个超短 CPU 跑里收敛），并如实记录上面的观测结论——这本身构成了 11.3 的
"对比"：在同一合成任务上，DPO 与 GRPO 都是这里 CPU 可验证的、稳定收敛的选择，而 PPO
的收敛需要更大 rollout 预算 / 更稠密的奖励。

## 测试

`tests/training/test_aligner_benchmark.py` 覆盖：`TargetTokenReward` 的打分与 mask；
`preference_fraction` 的排序与形状校验；DPO 基准 e2e（偏好比例 0 -> ~1）；GRPO 基准
e2e（group-reward fraction 0 -> >0.9）；PPO 基准（真实 `PPOTrainer` 产出有限 reward
轨迹）；以及 `compare_dpo_vs_ppo` 三方对比返回 summary 的端到端入口。
