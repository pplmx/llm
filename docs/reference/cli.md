# CLI 命令参考

所有命令行工具的完整参数列表。

---

## train_simple_decoder.py

预训练模型脚本。

### 位置

```bash
scripts/train_simple_decoder.py
```

### 参数

#### 必需参数

| 参数          | 类型 | 说明             |
| ------------- | ---- | ---------------- |
| `--file-path` | Path | 训练数据文件路径 |

#### 模型参数

| 参数            | 默认值 | 说明             |
| --------------- | ------ | ---------------- |
| `--hidden-size` | 64     | 隐藏层维度       |
| `--num-layers`  | 2      | Transformer 层数 |
| `--num-heads`   | 2      | 注意力头数       |
| `--max-seq-len` | 32     | 最大序列长度     |

#### 训练参数

| 参数             | 默认值 | 说明            |
| ---------------- | ------ | --------------- |
| `--batch-size`   | 16     | 批次大小        |
| `--epochs`       | 1      | 训练轮数        |
| `--lr`           | 1e-3   | 学习率          |
| `--device`       | auto   | 设备 (cpu/cuda) |
| `--log-interval` | 10     | 日志输出间隔    |

#### Checkpoint 参数

| 参数              | 默认值        | 说明                   |
| ----------------- | ------------- | ---------------------- |
| `--save-dir`      | ./checkpoints | 保存目录               |
| `--save-interval` | 100           | 保存间隔 (步数)        |
| `--resume`        | None          | 恢复的 checkpoint 路径 |

#### 早停参数

| 参数                         | 默认值 | 说明         |
| ---------------------------- | ------ | ------------ |
| `--val-file-path`            | None   | 验证数据文件 |
| `--early-stopping-patience`  | 3      | 早停耐心值   |
| `--early-stopping-min-delta` | 0.001  | 最小改善阈值 |

### 使用示例

```bash
# 基本使用
python scripts/train_simple_decoder.py --file-path data.txt

# 带 checkpoint
python scripts/train_simple_decoder.py \
    --file-path data.txt \
    --save-dir ./checkpoints \
    --save-interval 50

# 从 checkpoint 恢复
python scripts/train_simple_decoder.py \
    --file-path data.txt \
    --resume ./checkpoints/latest.pt
```

---

## 训练配置 (YAML)

使用配置文件进行复杂训练：

```yaml
model:
  hidden_size: 256
  num_layers: 4
  num_heads: 4
  vocab_size: 32000

training:
  batch_size: 32
  epochs: 3
  lr: 1e-4

optimization:
  use_amp: true
  use_compile: true

distributed:
  backend: nccl
  world_size: 4
```

---

## llm-quantize

模型量化 CLI,目前仅支持 `gptq` 子命令 (Frantar 2022 Hessian-aware 4/8-bit PTQ)。

### 位置

`src/llm/cli/quantize.py`,通过 `pyproject.toml` 的 `[project.scripts]` 注册为 `llm-quantize`。

### 用法

```bash
llm-quantize gptq \
    --model PATH                 # torch.save blob (含 DecoderModel) \
    --output PATH                # 量化模型输出路径 \
    --calib-data PATH            # 原始文本 (每行一个样本) — 需搭配 --tokenizer \
    --calib-data-tokens PATH     # 预分词 .pt 文件 — 与 --calib-data 互斥 \
    --tokenizer PATH             # HF tokenizer 目录; 与 --calib-data 同时使用 \
    --bits {4,8}                 # 默认 4 \
    --group-size N|-1            # 默认 128; -1 = per-channel \
    [--sym|--asym]               # 默认 sym (4-bit packed storage 假设 sym) \
    [--act-order|--no-act-order] # 默认 off \
    --percdamp F                 # 默认 0.01 \
    --blocksize N                # 默认 128 \
    --target-modules m1,m2,...   # 默认所有 nn.Linear
```

### 退出码

| 码 | 含义 |
|----|------|
| 0  | 量化成功 |
| 1  | 参数校验失败 (`--bits` 非法 / 缺 `--tokenizer` / `--model` 不存在等) |
| 2  | 运行失败 (torch.load 失败 / 分词失败 / 量化内核异常 / 保存失败) |

### 校验规则 (失败即退出码 1)

- `--bits` 必须为 4 或 8
- `--group-size` 必须为 -1 (per-channel) 或正整数
- `--percdamp` 必须 ∈ (0, 1)
- `--blocksize` 必须为正,且当 `--group-size > 0` 时必须能被 `--group-size` 整除
- `--calib-data` 与 `--calib-data-tokens` **互斥**,必须二选一
- `--calib-data` 必须搭配 `--tokenizer` (原始文本需要分词)
- `--model` 必须存在且为常规文件

### 使用示例

```bash
# 用 HF tokenizer 分词原始文本
llm-quantize gptq \
    --model ckpt.pt \
    --output ckpt-int4.pt \
    --calib-data calibration_texts.txt \
    --tokenizer gpt2 \
    --bits 4 \
    --group-size 128 \
    --act-order

# 用预分词 .pt 文件 (无需 --tokenizer)
llm-quantize gptq \
    --model ckpt.pt \
    --output ckpt-int8.pt \
    --calib-data-tokens calib_tokens.pt \
    --bits 8

# 只量化指定层,其余保持 fp32
llm-quantize gptq \
    --model ckpt.pt \
    --output ckpt-mixed.pt \
    --calib-data-tokens calib_tokens.pt \
    --target-modules fc1,fc2 \
    --bits 4
```

### 与 Python API 的关系

`llm-quantize gptq` 是 `llm.quantization.gptq.quantize_model_gptq` 的薄包装。
所有量化算法参数 (Hessian 阻尼、列块大小、act-order 等) 直接映射到 `GPTQConfig`
的字段 — Python 端的 `GPTQConfig.__post_init__` 校验仍会执行,作为 defense-in-depth
兜底。CLI 端提前校验只为给用户一个清晰的一行错误信息,而不是堆栈帧。

---

## 环境变量

| 变量                     | 说明          |
| ------------------------ | ------------- |
| `CUDA_VISIBLE_DEVICES`   | 可见 GPU 设备 |
| `NCCL_DEBUG`             | NCCL 调试级别 |
| `TORCH_CUDNN_V8_ENABLED` | cuDNN v8 优化 |
