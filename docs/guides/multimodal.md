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

## 已落地：模态融合模型 + 训练任务（slice 2）

- `MultimodalModel`（`llm/multimodal/model.py`，独立模型，**不 patch DecoderModel**）：
  在 token-embedding 空间把 registry 的 `modal_embeds` 作为 prefix 前缀注入：把
  `modality_fusion(modal_embeds)` 拼到文本 embedding 前，跑 decoder 的 transformer
  blocks + LM head，返回**文本** logits。推荐 `use_rope=True`（位置在 attention 内注入，
  不受 `max_seq_len` 的加法位置表限制）。
- `--task multimodal`（`MultimodalTask` + `MultimodalDataModule`，已注册）：标准训练
  循环，batch 携带 `modal_embeds`，以 CE 优化文本 next-token。

```python
from llm.multimodal import MultimodalDataModule, MultimodalTask
from llm.training.core.engine import TrainingEngine

module = MultimodalDataModule(config, modality="linear", input_dim=16)  # use_rope=True
task = MultimodalTask(config, module)
engine = TrainingEngine(config=config, task=task, rank=0, world_size=1, data_module=module)
loss = engine._run_epoch(0)   # 批次含 input_ids / labels / modal_embeds
```

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
- 真实视觉/音频编码器 + 图像-文本对齐 / Visual Instruction Tuning（ROADMAP 阶段十二
  的 12.1/12.2/12.3）：当前模型/任务切片用合成 `linear` 模态验证流程，真实视觉/音频
  注册即可接入，不改模型核心。
- 真实多模态数据集接入。

测试见 `tests/multimodal/`（registry + DataModule 契约 + 最小编码器可训练性）。
