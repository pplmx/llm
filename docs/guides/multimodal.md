# 多模态扩展 (Multimodal) — 契约 spike

多模态是 ROADMAP 阶段十二（P2）与"下一步探索方向 P1"的扩展方向。首个落地是
**契约优先的 spike**：先定义数据 + 编码器扩展面，纯 CPU 可验证，**不硬改
`DecoderModel`**（架构边界见
[ADR-013](/llm/docs/adr/013-multimodal-encoder-contract.md)）。

## 已落地：数据 / 编码器契约

- `ModalityEncoder`（`abc.ABC, nn.Module`）：一个模态编码器把该模态的原始样本编码成
  固定 `embed_dim` 的 embedding。
- `MODALITY_ENCODER_REGISTRY` + `@register_encoder(name)`：与 `SOURCE_REGISTRY` /
  `PEFT_REGISTRY` 同构，支持第三方用 `llm.modality_encoders` entry point 扩展。
- 内置 `LinearModalityEncoder`：`input_dim -> embed_dim` 线性投影（可训练、无外部依赖、
  CPU 可验证），作为真实编码器接入前的占位。
- `MultimodalDataModule`：把文本 token（next-token labels）与一个辅助模态特征配对，
  经 registry 编码器产出 `modal_embeds`；batch 契约：
  `{"input_ids": [B,T], "labels": [B,T], "modal_embeds": [B, embed_dim]}`。

```python
from llm.multimodal import MODALITY_ENCODER_REGISTRY, MultimodalDataModule

enc = MODALITY_ENCODER_REGISTRY.get("linear")(input_dim=16, embed_dim=24)
embeds = enc(feature)                      # [N, 24]

module = MultimodalDataModule(config, modality="linear", input_dim=16)
module.setup()
batch, _ = module.train_dataloader(rank=0, world_size=1)
# batch["input_ids"], batch["labels"], batch["modal_embeds"]
```

## 未落地（后续切片）

- 真实视觉/音频编码器（CLIP/SigLIP 等）——注册即可接入，不改模型核心。
- `MultimodalModel`（模态 tokenizer + 融合层 + 训练任务）：届时单独设计，仍不改
  `DecoderModel`（除非新 ADR 批准边界变更）。
- 真实多模态数据集接入。

测试见 `tests/multimodal/`（registry + DataModule 契约 + 最小编码器可训练性）。
