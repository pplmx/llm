# RLAIF (RL from AI Feedback)

RLAIF 用 **AI/规则 judge** 代替人类来标注偏好，再走既有对齐流程训练。本切片把 judge
标定的 (chosen, rejected) 响应对喂给现有的 **DPO 任务**，是 ROADMAP 阶段十一
「其他对齐技术」的一部分，与 GRPO / Rejection Sampling 互补。

## 组件

- `llm/training/rlhf/aifeedback.py`：
    - `PreferenceJudge`（ABC）：给每条 response 打分（`score_batch`）。
    - `TargetTokenJudge`：规则/scalar judge——响应以目标 token 结尾则得分 1，否则 0
  （CPU 可验证，作为真实 judge 模型的占位）。
    - `prefer_batch(a, b, judge)`：把两条 batch 按分数排序成 `(chosen, rejected)`。
- `llm/data/modules/aifeedback.py`：`AIFeedbackDataModule`——合成响应对，用 judge
  标注，产出 DPO 兼容 batch
  `{chosen_input_ids, chosen_labels, chosen_attention_mask, rejected_* }`，
  可直接接入现有 `--task dpo`。

## 用法

```python
from llm.data.modules.aifeedback import AIFeedbackDataModule
from llm.training.tasks.dpo_task import DPOTask
from llm.training.core.engine import TrainingEngine

module = AIFeedbackDataModule(config)  # 默认 TargetTokenJudge
task = DPOTask(config, module)
engine = TrainingEngine(config=config, task=task, rank=0, world_size=1, data_module=module)
loss = engine._run_epoch(0)
```

## 测试

`tests/training/test_aifeedback.py` 覆盖：judge 打分、`prefer_batch` 选择/形状校验、
DataModule 的 DPO batch 结构，以及一个 CPU e2e——DPO 在 AI 标注的偏好对上训练后，
策略对 judge 选中响应的偏好比例从 ~0 升到 ~1。
