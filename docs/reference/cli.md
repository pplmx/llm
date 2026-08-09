---
tags:
  - 参考
  - CLI
---

# CLI 命令参考

所有注册命令行工具的完整参数列表（`pyproject.toml` `[project.scripts]` 注册）：

| 命令              | 用途                                             |
| ----------------- | ------------------------------------------------ |
| `llm-train`       | 训练（预训练 / SFT / DPO / reward / PPO / demo） |
| `llm-serve`       | OpenAI 兼容推理服务                              |
| `llm-quantize`    | GPTQ 量化                                        |
| `llm-migrate-ckpt`| 旧版 checkpoint 迁移到 v2 split 布局             |

> `scripts/` 下的 demo 脚本（如 `train_simple_decoder.py`）不是注册 CLI，仅作最小演示；生产请走 `llm-train`。

---

## llm-train

统一训练 CLI（入口 `llm.training.train:app`）。**注意：任务名通过 `--task` 传入，不是子命令。**

### 用法

```bash
llm-train --task <task> [OPTIONS]
```

### 任务（`--task`，必填）

| 任务        | 用途                                                    |
| ----------- | ------------------------------------------------------- |
| `lm`        | Map-style 语言建模（`TextDataModule`）                  |
| `stream_lm` | 流式大规模预训练（`StreamingTextDataModule`，主入口）   |
| `sft`       | 监督微调                                                |
| `dpo`       | 直接偏好优化                                            |
| `reward`    | Reward model 训练                                       |
| `ppo`       | PPO RLHF 对齐                                           |
| `regression`| 合成回归 demo                                           |

### 参数

| 参数                       | 说明                                         |
| -------------------------- | -------------------------------------------- |
| `--task`                   | 任务名（必填，choices 来自 `TASK_REGISTRY`） |
| `--config-path`            | YAML 配置文件路径                            |
| `--epochs`                 | 覆盖训练轮数                                 |
| `--batch-size`             | 覆盖 batch size                              |
| `--lr`                     | 覆盖学习率                                   |
| `--num-samples`            | 覆盖合成样本数                               |
| `--steps-per-epoch`        | 覆盖流式每 epoch 步数                        |
| `--compile`/`--no-compile` | torch.compile（默认开启）                    |
| `--amp`/`--no-amp`         | 混合精度（默认开启）                         |

> resume、PEFT、checkpoint 目录等**没有 CLI 参数**，统一通过 YAML 配置：`checkpoint.resume_from_checkpoint` / `checkpoint.checkpoint_dir` / `training.peft_method` / `training.peft_kwargs` / `training.peft_save_path` 等。

### 示例

```bash
# 流式预训练（本地冒烟，CPU 可跑）
uv run llm-train --task stream_lm --config-path configs/streaming_local_demo.yaml

# 流式预训练（C4，生产规模）
uv run llm-train --task stream_lm --config-path configs/streaming_c4.yaml

# SFT + LoRA：PEFT 走 YAML（training 段），不是 CLI 参数
uv run llm-train --task sft --config-path configs/sft_alpaca.yaml

# CLI 覆盖实验参数
uv run llm-train --task sft --config-path configs/sft_alpaca.yaml \
  --epochs 3 \
  --batch-size 16 \
  --lr 2e-5
```

---

## llm-serve

推理服务 CLI（入口 `llm.serving.api:main`），启动 OpenAI 兼容 HTTP API。**只读环境变量，没有 `--config` 参数。**

### 用法

```bash
llm-serve [OPTIONS]   # 实际参数全部来自 LLM_SERVING_* 环境变量
```

### 环境变量配置

#### 模型与推理

| 变量                                  | 默认值 | 说明                                                                                                                       |
| ------------------------------------- | ------ | -------------------------------------------------------------------------------------------------------------------------- |
| `LLM_SERVING_MODEL_PATH`              | None   | 训练 checkpoint：v2 三件套的 stem（或 `.safetensors` 路径），也接受旧式单文件 `.pt`；None = dummy 模型                     |
| `LLM_SERVING_TOKENIZER_PATH`          | None   | tokenizer pickle 或 HF repo id                                                                                             |
| `LLM_SERVING_TOKENIZER_TYPE`          | simple | `simple` / `hf`                                                                                                            |
| `LLM_SERVING_DEVICE`                  | auto   | 推理设备                                                                                                                   |
| `LLM_SERVING_GENERATION_BACKEND`      | eager  | `eager` / `batched`（speculative 需要 target + draft 双模型，走 Python API，见 [Inference Guide](../guides/inference.md)） |
| `LLM_SERVING_COMPILE_MODEL`           | false  | 启动时 torch.compile                                                                                                       |
| `LLM_SERVING_MAX_CONCURRENT_REQUESTS` | 4      | 并发请求上限（semaphore）                                                                                                  |
| `LLM_SERVING_REQUEST_TIMEOUT`         | 60.0   | 单请求超时（秒）                                                                                                           |

#### 安全与可观测

| 变量                    | 默认值    | 说明                                                                             |
| ----------------------- | --------- | -------------------------------------------------------------------------------- |
| `LLM_SERVING_HOST`      | 127.0.0.1 | 监听地址；绑定非回环地址时**必须**同时设置 `LLM_SERVING_API_KEY`（否则启动失败） |
| `LLM_SERVING_API_KEY`   | None      | API 密钥；`None` + 非回环 host 会被拒绝启动                                      |
| `LLM_SERVING_LOG_LEVEL` | INFO      | 日志级别                                                                         |
| `LLM_SERVING_RELOAD`    | 关        | uvicorn 自动重载（`1`/`true`/`yes`，仅本地开发）                                 |

#### KV cache / Paged Attention / Prefix Cache

| 变量                              | 默认值 | 说明                         |
| --------------------------------- | ------ | ---------------------------- |
| `LLM_SERVING_USE_PAGED_ATTENTION` | false  | 启用 block-allocator KV      |
| `LLM_SERVING_MAX_BLOCKS`          | 256    | 最大 block 数                |
| `LLM_SERVING_BLOCK_SIZE`          | 16     | block 大小                   |
| `LLM_SERVING_ENABLE_PREFIX_CACHE` | false  | 多轮 chat 摊销 system prompt |
| `LLM_SERVING_MAX_PREFIXES`        | 10     | 缓存的前缀条数               |

#### Chat template（`/v1/chat/completions`）

| 变量                                 | 默认值 | 说明                                          |
| ------------------------------------ | ------ | --------------------------------------------- |
| `LLM_SERVING_CHAT_MESSAGE_TEMPLATE`  | None   | 消息渲染格式（占位符 `{role}` / `{content}`） |
| `LLM_SERVING_CHAT_GENERATION_PREFIX` | None   | 消息末尾追加的生成前缀（默认 `Assistant: `）  |

#### PEFT adapter（训练 → 服务闭环）

| 变量                            | 默认值 | 说明                                                                                                         |
| ------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------ |
| `LLM_SERVING_PEFT_METHOD`       | None   | 方法名（`lora` / `ia3` / `bitfit` / `adapter` / `pfeiffer_adapter` / `adalora` / `qlora` / `prefix_tuning`） |
| `LLM_SERVING_PEFT_KWARGS`       | {}     | 传给 `apply_peft` 的 kwargs                                                                                  |
| `LLM_SERVING_PEFT_ADAPTER_PATH` | None   | `save_peft` 写的 sidecar 路径                                                                                |
| `LLM_SERVING_PEFT_MERGE`        | false  | 启动时 merge 进 base 权重（换取吞吐，失去运行时 swap 能力）                                                  |

#### dummy 模型架构（`model_path` 为空时使用；加载 checkpoint 后以 checkpoint 的 `model_config` 为准）

| 变量                       | 默认值 | 说明                       |
| -------------------------- | ------ | -------------------------- |
| `LLM_SERVING_HIDDEN_SIZE`  | 64     | 隐藏层维度                 |
| `LLM_SERVING_NUM_LAYERS`   | 2      | Transformer 层数           |
| `LLM_SERVING_NUM_HEADS`    | 4      | 注意力头数                 |
| `LLM_SERVING_MAX_SEQ_LEN`  | 128    | 最大序列长度               |
| `LLM_SERVING_NUM_KV_HEADS` | None   | GQA KV 头数                |
| `LLM_SERVING_NUM_EXPERTS`  | 0      | MoE 专家数（0 = 关闭）     |
| `LLM_SERVING_TOP_K`        | 0      | MoE top-k                  |
| `LLM_SERVING_ATTN_IMPL`    | mha    | 注意力实现（mha/mla）      |
| `LLM_SERVING_MLP_IMPL`     | mlp    | MLP 实现（mlp/moe/swiglu） |

### 示例

```bash
# 基础服务（dummy 模型，smoke test）
uv run llm-serve

# 带 checkpoint 和 HF tokenizer
LLM_SERVING_MODEL_PATH=checkpoints/epoch_5 \
LLM_SERVING_TOKENIZER_PATH=gpt2 \
LLM_SERVING_TOKENIZER_TYPE=hf \
uv run llm-serve

# 生产部署（Paged Attention + API key）
LLM_SERVING_HOST=0.0.0.0 \
LLM_SERVING_API_KEY=$(openssl rand -hex 32) \
LLM_SERVING_USE_PAGED_ATTENTION=true \
LLM_SERVING_GENERATION_BACKEND=batched \
uv run llm-serve
```

### 公开主机守卫

绑定非回环地址（`0.0.0.0`、公网 IP 等）且未配置 `LLM_SERVING_API_KEY` 时，`llm-serve` **拒绝启动**（fail-closed），避免匿名暴露推理端点。

---

## llm-quantize

模型量化 CLI，目前支持 `gptq` 子命令（Frantar 2022 Hessian-aware 4/8-bit PTQ）。
方法选型、Python API 与质量验证见[模型量化指南](../guides/quantization.md)。

### 用法

```bash
llm-quantize gptq \
    --model PATH                 # torch.save blob（含 DecoderModel）\
    --output PATH                # 量化模型输出路径 \
    --calib-data PATH            # 原始文本（每行一个样本）— 需搭配 --tokenizer \
    --calib-data-tokens PATH     # 预分词 .pt 文件 — 与 --calib-data 互斥 \
    --tokenizer PATH             # HF tokenizer 目录；与 --calib-data 同时使用 \
    --bits {4,8}                 # 默认 4 \
    --group-size N|-1            # 默认 128；-1 = per-channel \
    [--sym|--asym]               # 默认 sym（4-bit packed storage 假设 sym）\
    [--act-order|--no-act-order] # 默认 off \
    --percdamp F                 # 默认 0.01 \
    --blocksize N                # 默认 128 \
    --target-modules m1,m2,...   # 默认所有 nn.Linear
```

### 退出码

| 码  | 含义                                                                  |
| --- | --------------------------------------------------------------------- |
| 0   | 量化成功                                                              |
| 1   | 参数校验失败（`--bits` 非法 / 缺 `--tokenizer` / `--model` 不存在等） |
| 2   | 运行失败（torch.load 失败 / 分词失败 / 量化内核异常 / 保存失败）      |

### 校验规则（失败即退出码 1）

- `--bits` 必须为 4 或 8
- `--group-size` 必须为 -1（per-channel）或正整数
- `--percdamp` 必须 ∈ (0, 1)
- `--blocksize` 必须为正，且当 `--group-size > 0` 时必须能被 `--group-size` 整除
- `--calib-data` 与 `--calib-data-tokens` **互斥**，必须二选一
- `--calib-data` 必须搭配 `--tokenizer`（原始文本需要分词）
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

# 用预分词 .pt 文件（无需 --tokenizer）
llm-quantize gptq \
    --model ckpt.pt \
    --output ckpt-int8.pt \
    --calib-data-tokens calib_tokens.pt \
    --bits 8

# 只量化指定层，其余保持 fp32
llm-quantize gptq \
    --model ckpt.pt \
    --output ckpt-mixed.pt \
    --calib-data-tokens calib_tokens.pt \
    --target-modules fc1,fc2 \
    --bits 4
```

### 与 Python API 的关系

`llm-quantize gptq` 是 `llm.quantization.gptq.quantize_model_gptq` 的薄包装。
所有量化算法参数（Hessian 阻尼、列块大小、act-order 等）直接映射到 `GPTQConfig`
的字段 — Python 端的 `GPTQConfig.__post_init__` 校验仍会执行，作为 defense-in-depth
兜底。CLI 端提前校验只为给用户一个清晰的一行错误信息，而不是堆栈帧。

---

## llm-migrate-ckpt

将 v0.0.5 及更早的旧式单文件 `.pt` checkpoint 转换为 v2 split 三件套
（`<stem>.safetensors` + `<stem>.meta.json` + `<stem>.extra_state.pt`）。
转换是**卫生性**操作：`CheckpointManager` 的 loader 本身兼容两种布局，迁移只为了让新代码路径（safetensors 部分加载、HF 发布等）可直接使用。

### 用法

```bash
llm-migrate-ckpt [OPTIONS] {path}
```

`path` 可以是 `<name>.pt` 或 stem `<name>`（自动补 `.pt`）。

### 参数

| 参数          | 说明                                                                     |
| ------------- | ------------------------------------------------------------------------ |
| `--in-place`  | 转换成功后删除旧 `.pt`（默认保留，便于先验证）                           |
| `--verify`    | 转换后重新加载新旧布局并对比（权重 float 漂移容差 1e-5）；不一致退出码 2 |
| `--dry-run`   | 只打印转换计划，不写入任何文件                                           |
| `--overwrite` | 覆盖同 stem 已存在的 split 三件套（默认拒绝，防误覆盖）                  |

### 示例

```bash
# 基本转换（保留旧 .pt）
llm-migrate-ckpt checkpoints/epoch_5.pt

# 转换 + 校验 + 删除旧文件
llm-migrate-ckpt checkpoints/epoch_5 --verify --in-place

# 只预览会做什么
llm-migrate-ckpt checkpoints/epoch_5 --dry-run
```

---

## 环境变量

| 变量                     | 说明          |
| ------------------------ | ------------- |
| `CUDA_VISIBLE_DEVICES`   | 可见 GPU 设备 |
| `NCCL_DEBUG`             | NCCL 调试级别 |
| `TORCH_CUDNN_V8_ENABLED` | cuDNN v8 优化 |
