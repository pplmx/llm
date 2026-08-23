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

## 已落地：CLIP/SigLIP 风格视觉编码器 slice 1（ROADMAP 12.1）

- `VisionTransformerEncoder`（`llm/multimodal/vision.py`，注册为 `"vit"`）：
  消费**原始图像** `[B, C, H, W]`（而非预计算特征向量），输出 ViT 布局的
  image-token embeddings `[B, num_tokens, embed_dim]`。架构（ViT-B 风格、纯 CPU
  可验证）：`ImagePatchPreprocessor`（12.3 patchify + 线性投影 + 可学习位置编码）→
  可选可学习 `[CLS]` token（CLIP 式，置 0 位）→ N 个 pre-norm transformer block
  （LayerNorm → 多头 SDPA → residual → LayerNorm → MLP(GELU, ×mlp_ratio) →
  residual）→ 最终 LayerNorm（SigLIP/CLIP 式）。
- `with_cls` 控制是否前置 `[CLS]`（`num_tokens = N` 或 `N+1`）；`freeze_encoder=True`
  冻结整座塔（CLIP 常见做法）。`MultimodalDataModule(modality="vit", ...)` 生成合成
  图像，在 setup 时经冻结塔产出 `modal_embeds [B, num_tokens, embed_dim]`（3D，
  与 linear 路径的 2D `[B, embed_dim]` 不同），直接喂 `MultimodalModel` 的 prefix
  融合，图像→token→decoder 前缀的完整训练循环在 CPU 上收敛。

```python
from llm.multimodal import MultimodalDataModule, MultimodalTask
from llm.training.core.engine import TrainingEngine

module = MultimodalDataModule(config, modality="vit", image_h=64, image_w=64,
                              patch_size=16, vit_layers=2, vit_heads=4, with_cls=True)
module.setup()
assert module.num_modal_tokens == 17      # 16 patches (+ CLS)
task = MultimodalTask(config, module)
engine = TrainingEngine(config=config, task=task, rank=0, world_size=1, data_module=module)
loss = engine._run_epoch(0)               # batch["modal_embeds"]: [B, 17, embed_dim]
```

## 已落地：图像-文本对齐模块 slice 3（ROADMAP 12.1）

- `ContrastiveAligner`（`llm/multimodal/alignment.py`）：CLIP/SigLIP 风格对比对齐
  头。对图像 token 与文本 token 各做线性投影到共享空间并 L2 归一化,以可学习温度
  `scale = exp(logit_scale)` 打分 `logits[B,B]`,损失为对称 InfoNCE(或 SigLIP
  sigmoid 变体 `sigmoid=True`)。图像侧支持 `image_pool='mean'`(所有 token 均值)或
  `'cls'`(首行 `[CLS]`)。纯 CPU 可验证:随机配对数据上损失从 `log(B)` 收敛,图像→
  文本 top-1 检索准确率升至 ~1;直接消费 `VisionTransformerEncoder` 输出(`[B,N,D]`)。
- 视觉塔**在线训练**已落地 slice 2: `MultimodalDataModule(..., train_encoder=True)`
  batch 携带原始图像 `images [B,3,H,W]`,`MultimodalModel`/`MultimodalTask` 在
  forward 内实时编码(视觉塔 → projector → 文本前缀联合训练),梯度可到达视觉塔;
  默认 `train_encoder=False` 保持冻结-预计算路径。

## 未落地（后续切片）

- Visual Instruction Tuning（ROADMAP 12.1 后续）——基于本 slice 的图像 token 输出
  与对齐头即可接入。
- 音频编码器（Whisper-style，ROADMAP 12.2）与真实多模态数据集接入。

测试见 `tests/multimodal/`（registry + DataModule 契约 + 最小编码器可训练性 +
视觉编码器形状/parity/梯度/冻结 + 图像路径 e2e 收敛）。
