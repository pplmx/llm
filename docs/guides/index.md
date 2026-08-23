# 功能指南

按使用场景组织的深度指南。如果你是第一次接触框架，建议先走
[🚀 快速开始](../getting-started.md)，再按下面的分类深入。

<div class="grid cards" markdown>

- 🚀 **训练与数据**

    ---

    - [数据管线](data.md) — 数据源、`SOURCE_REGISTRY`、C4 / Pile / RedPajama 预设
    - [流式预训练](streaming.md) — `stream_lm` 全流程、dedup、resume cursor
    - [Checkpoint](checkpoints.md) — v2 split 三件套、保存 / 恢复 / 迁移
    - [分布式训练](distributed.md) — DDP 与 FSDP、多节点启动

- 🧠 **模型与微调**

    ---

    - [PEFT 微调](finetuning.md) — LoRA / QLoRA / AdaLoRA / IA³ / BitFit / Adapter / Prefix Tuning
    - [模型量化](quantization.md) — GPTQ / AWQ / SmoothQuant / 混合精度
    - [知识蒸馏](distillation.md) — `--task distill`、teacher-from-checkpoint、温度缩放 KL

- ⚡ **推理与部署**

    ---

    - [推理优化](inference.md) — KV cache、Paged Attention、连续批处理、Flash Attention
    - [模型导出](export.md) — ONNX / TorchScript / GGUF、自定义后端
    - [模型评估](evaluation.md) — lm-evaluation-harness、MMLU / ARC / WikiText

</div>

## 相关入口

| 文档                                             | 说明                         |
| ------------------------------------------------ | ---------------------------- |
| [预训练教程](../tutorials/01-pretraining.md)     | 从零训练一个小型语言模型     |
| [微调教程](../tutorials/02-finetuning.md)        | SFT / DPO + LoRA/QLoRA 实操  |
| [推理教程](../tutorials/03-inference.md)         | `llm-serve` 部署与 API 调用  |
| [系统架构](../reference/architecture.md)         | 分层设计、Registry 机制      |
| [CLI 命令参考](../reference/cli.md)              | `llm-train` / `llm-serve` 等 |
