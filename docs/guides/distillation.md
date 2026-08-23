# 知识蒸馏 (Knowledge Distillation)

知识蒸馏把一个大/已训练模型（**teacher**）的"软"预测迁移到一个更紧凑的
学生模型（**student**），是一种模型压缩手段（ROADMAP 13.4，RIL DEC-055）。
本框架用 Hinton 温度缩放 KL 实现，入口是训练任务 `distill`。

## 目标损失

对每个 token，损失为：

```text
loss = alpha * CE(student, y)
     + (1 - alpha) * T^2 * KL(softmax(student/T) || softmax(teacher/T))
```

- `T`（`distill_temperature`）：软化温度，越高目标越"软"、梯度越平滑。
- `alpha`（`distill_alpha`）：hard-label CE 权重；KL 项贡献 `(1-alpha)`。
- 只有 student 的参数量被优化；teacher 完全冻结（`requires_grad=False`）。

实现见 `llm/training/distillation.py` 与 `llm/training/tasks/distill_task.py`。

## 用法

`distill` 任务的 teacher **从 checkpoint 加载**（必须与 `config.model`
同架构：vocab / hidden / layers / heads 一致，state dict 才能直接装载）。
teacher 用任何 `lm` 系列任务训练后由 `CheckpointManager` 保存即可。

在 `TrainingConfig` 里配置：

```yaml
training:
  task: distill
  distill_teacher_path: /path/to/teacher/ckpt   # CheckpointManager 保存的 stem（如 .../best）
  distill_temperature: 4.0
  distill_alpha: 0.5
  lr: 1e-3
```

然后：

```bash
uv run llm-train --task distill --config-path configs/your_distill.yaml
```

## 端到端步骤

1. 用 `lm` / `stream_lm` 训练一个模型并保存 checkpoint（得到 teacher）。
2. 新建 student 配置：把 `distill_teacher_path` 指向该 checkpoint，并设置温度 / alpha。
3. 跑 `--task distill`。student 以随机初始化开始，逐步逼近 teacher 的软输出。

## CPU / 测试

`tests/training/test_distill_task.py` 提供一个纯 CPU、可复现的演示：在固定的小词汇
语料上先预训练一个 teacher（保证其输出非 uniform、有可学习的结构），保存成
checkpoint，再蒸馏一个全新 student，断言其 KD loss 与 KL-to-teacher 都显著下降
（student 确实向 teacher 靠近）。也可在无 `distill_teacher_path` 时快速体验：
此时框架构造一个冻结的随机 teacher（仅用于开发 / 冒烟，非真实蒸馏）。
