# 图像→Token 预处理器（研究切片）

阶段 12.3「实现模态特定的预处理器」。多模态 spike（`multimodal/encoders.py` +
`data.py`）用线性 encoder 当真实视觉编码器的占位，直接消费 raw 特征向量——缺了真正的
**图像→token** 一步。本切片补上：ViT 风格的图像预处理器，CPU 可验证。

## 用法

```python
from llm.multimodal.preprocess import patchify, ImagePatchPreprocessor

img = torch.randn(2, 3, 32, 32)  # [B, C, H, W]
patches = patchify(img, patch_size=16)  # [B, N, C*p*p]  (N = (H/p)(W/p))
pp = ImagePatchPreprocessor(in_channels=3, patch_size=16, embed_dim=8, image_h=32, image_w=32)
tokens = pp(img)  # [B, N, embed] 图像 token 嵌入
```

`patchify` 用 `unfold` 把图像切成方 patch；`ImagePatchPreprocessor` 把每个 patch
flatten → learned linear 投影 → 加上 ViT 式可学习 positional embedding，输出标准
`[B, N, embed]` 图像 token 序列，可直接喂给多模态主干。独立切片，不 patch
`DecoderModel`；把真实的 CLIP/SigLIP encoder 接进来列为后续。

## CPU parity 不变量（见 `tests/multimodal/test_preprocess.py`）

- `patchify` === 显式手工切 patch 参考；
- 输出 `[B, N, embed]` 且确定（同输入同输出）；
- `proj` 与 `pos_embed` 的 backward 梯度有限；
- 尺寸不可被 `patch_size` 整除 / patch 数超过 pos_embed 槽位时抛清晰 `ValueError`。
