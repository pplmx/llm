# 013. Multimodal encoder & data-module contract (spike)

Date: 2026-08-23

## Status

Accepted (spike slice - ROADMAP 阶段十二 / RIL DEC-058 / TASK-226)

## Context

多模态是 ROADMAP 阶段十二（P2）并位列"下一步探索方向 P1"。仓库架构边界明确要求：
多模态无 registry，需先设计 MultimodalDataModule 等，勿硬改 DecoderModel。因此在接入
真实视觉/音频模型之前，需要先落定数据 + 编码器的契约面，且必须能在纯 CPU 上验证。

现有类似扩展面（SOURCE_REGISTRY / PEFT_REGISTRY / MODEL_REGISTRY）都基于
llm.runtime.registry.Registry 加 entry-point 扩展；DecoderModel 是单一 text-only
结构，硬加模态分支会破坏既有路径。

## Decision

契约优先的 spike 切片：

1. llm/multimodal/encoders.py：ModalityEncoder（abc.ABC, nn.Module，定义 modality /
   encode(sample) -> Tensor / embed_dim）；MODALITY_ENCODER_REGISTRY: Registry[type] +
   @register_encoder(name)；load_entry_point_registry("llm.modality_encoders", ...) 支持
   第三方扩展；内置最小纯 CPU 可验证的 LinearModalityEncoder（input_dim -> embed_dim
   线性投影，可训练）。
2. llm/multimodal/data.py：MultimodalDataModule（SamplerMapDataModule 子类）把文本
   token（next-token labels）与辅助模态特征配对，经 registry 编码器产出 modal_embeds；
   batch 契约 {"input_ids": [B,T], "labels": [B,T], "modal_embeds": [B, embed_dim]}。
3. 不触碰 DecoderModel：本切片只在数据/编码器层注册扩展面，模型集成留后续独立切片。
4. llm.multimodal 加入序列化安全白名单（serialization._FRAMEWORK_PACKAGES），编码器
   未来可持久化 / 安全加载。

备选（拒绝）：在 DecoderModel 直接加模态分支（跨越架构边界，破坏 text-only 与 checkpoint
兼容）；引入真实图像模型（需外部权重/GPU，脱离 CPU 可验证范围）；复用 SOURCE_REGISTRY
（其语义是文本数据源，与模态编码器正交）。

## Consequences

- 先锁契约，后续视觉/音频编码器注册即可接入，无需改模型核心；纯 CPU 可验证；与既有
  Registry + entry-point 体系一致，第三方扩展零成本。
- 尚无真实模型消费 modal_embeds——需要后续"模态 tokenizer + 融合层 + 训练任务"切片，
  届时设计 MultimodalModel（仍不改 DecoderModel，除非新 ADR 批准边界变更）。
- MultimodalDataModule 目前是合成数据，真实图像/音频数据集接入是后续工作。
