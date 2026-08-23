# Rejection Sampling (Rejection Sampling Alignment)

Rejection sampling（best-of-N / top-K）是对齐的一种简单而有效的手段：对同一 prompt
采样多条 response，用奖励打分，**只保留奖励最高的子集**（top-K 或阈值以上），再对
这些高奖励 response 做监督微调（SFT-on-selected）。反复迭代即可让策略的奖励提升。
它是 ROADMAP 阶段十一「其他对齐技术」的一部分，与 GRPO 互补。

## 组件

- `llm/training/rlhf/rejection_sampling.py`：
    - `select_top_k(scores, k)`：保留得分最高的 k 条（k>N 时全保留）。
    - `select_above_threshold(scores, threshold)`：保留 `score >= threshold`。
    - `rejection_sample(scores, *, k | threshold)`：返回布尔掩码 + 统计
  （`kept_mean_reward` / `base_mean_reward` / `kept_fraction`），两者只能二选一。
- `llm/data/modules/rejection_sample.py`：`RejectionSampleDataModule`——合成响应
  （奖励=1 当且仅当响应以 token 0 结尾），按所选模式过滤后把**保留集**暴露为
  `{input_ids, labels}` 的 next-token 数据集，可直接接入 `lm` / `sft` 训练。

## 用法

```python
from llm.data.modules.rejection_sample import RejectionSampleDataModule
from llm.training.tasks.lm_task import LanguageModelingTask

module = RejectionSampleDataModule(config, mode="top_k", k=16)  # 或 mode="threshold", threshold=0.5
module.setup()
print(module.stats)  # 保留集平均奖励 > 原始平均奖励

task = LanguageModelingTask(config, module)  # 对保留集做 SFT
```

## 测试

`tests/training/test_rejection_sampling.py` 覆盖：top-K / threshold 选择正确性、`k>N`
钳制与非法参数、`rejection_sample` 二选一校验、保留集平均奖励 > 原始、以及一个 CPU e2e
——拒绝低奖励后对保留集 SFT，训练后保留响应的序列似然显著提升。
