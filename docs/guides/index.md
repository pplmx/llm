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
    - [量化参数搜索](quant_param_search.md) — 枚举 (bits, granularity) 最小化重建 MSE
    - [知识蒸馏](distillation.md) — `--task distill`、teacher-from-checkpoint、温度缩放 KL
    - [模型剪枝](pruning.md) — `llm-prune`、`PrunedLinear` weight_mask、magnitude/random 策略
    - [低秩分解](lowrank.md) — `llm-decompose`、`LowRankLinear` SVD U-V、rank 旋钮
    - [多模态扩展](multimodal.md) — `ModalityEncoderRegistry` + `MultimodalDataModule` 契约 spike
    - [图像→Token 预处理器](multimodal_preprocess.md) — ViT 风格 patchify、图像编码链路
    - [混合专家 (MoE)](expert_choice.md) — Expert Choice Routing（专家选 token、结构性负载均衡）
    - [Aux 负载均衡损失](load_balance.md) — Switch / ST-MoE 辅助均衡、防 gate 塌缩与 dead 专家
    - [Soft MoE](soft_moe.md) — slot 软分配、可微路由替代硬 top-k
    - [动态专家选择](dynamic_selection.md) — 按路由置信度逐 token 调节专家数
    - [GRPO](grpo.md) — 组相对策略优化（`--task grpo`、group advantage + 裁剪 policy loss）
    - [Rejection Sampling](rejection_sampling.md) — best-of-N / top-K 过滤高奖励响应再 SFT
    - [RLAIF](rlaif.md) — AI/规则 judge 标定偏好 → `--task dpo`
    - [SimPO](simpo.md) — 无参考奖励偏好优化（`--task simpo`、长度归一化隐式奖励）
    - [对齐方法对比](dpo_vs_ppo.md) — 同一合成偏好任务上的 DPO / PPO(RLHF) / GRPO 三方对比基准
    - [Constitutional AI](constitutional.md) — 规则 constitution + 自批判→改写切片
    - [Block Sparse Attention](block_sparse.md) — 块稀疏注意力掩码 + CPU parity
    - [StreamingLLM](streaming_llm.md) — attention-sink 流式掩码 + CPU parity
    - [Longformer](longformer.md) — dilated 滑窗注意力掩码 + CPU parity
    - [BigBird](big_bird.md) — global+window+random 掩码 + CPU parity
    - [Infinite Attention](infinite.md) — Infini-Attention 压缩记忆（非负特征映射 + 状态累积）
    - [统一稀疏注意力 API](sparse_attention.md) — `build_sparse_attention_mask(kind)` 派发

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
