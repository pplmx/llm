# FAQ - 常见问题

本文档收集了在使用 LLM 项目时的常见问题和解答.

## 目录

- [安装和环境](#安装和环境)
- [训练相关](#训练相关)
- [模型架构](#模型架构)
- [性能优化](#性能优化)
- [开发工具](#开发工具)

---

## 安装和环境

### Q: 如何设置开发环境？

使用 `make init` 命令即可一键设置：

```bash
make init
```

这会自动创建虚拟环境、安装依赖并配置 pre-commit 钩子.

详见：[Development Guide](development/README.md)

### Q: 为什么项目使用 `uv` 而不是 `pip`？

`uv` 是用 Rust 编写的现代 Python 包管理器, 相比 pip 有以下优势：

- **速度快**: 依赖解析和安装速度快 10-100 倍
- **可靠性**: 更好的依赖冲突解决
- **锁文件**: 提供 `uv.lock` 确保可重现构建

了解更多：[uv 官方文档](https://github.com/astral-sh/uv)

---

## 训练相关

### Q: 我应该使用哪个任务？

项目提供了两个主要任务：

- **`lm`**: 语言模型任务(推荐)- 用于训练生成式语言模型
- **`regression`**: 回归任务 - 用于简单的回归测试

示例：

```bash
# Regression task (uses synthetic data, works out of the box)
llm-train --task regression --epochs 10 --batch-size 32

# Language modeling task (requires dataset configuration)
llm-train --task lm --config-path configs/example.yaml --epochs 10
```

> [!NOTE]
> `--task lm` 需要通过配置文件指定数据集。快速实验可使用 `scripts/train_simple_decoder.py`。

### Q: 如何启用分布式训练？

使用 `torchrun` 启动多 GPU 训练：

```bash
torchrun --nproc_per_node=4 src/llm/training/train.py --task lm
```

详见：[DDP Deep Dive](development/deep-dive-ddp.md)

### Q: 内存不足 (OOM) 怎么办？

尝试以下方法：

1. **减小 batch size**: `--batch-size 16`
2. **启用混合精度**: 默认已启用 AMP
3. **减小模型大小**: `--model.hidden_size 512`
4. **使用 Gradient Checkpointing**: 将在未来版本中支持

详见：[Troubleshooting Guide](troubleshooting.md)

---

## 模型架构

### Q: 什么是 GQA (Grouped Query Attention)？

GQA 是一种优化的注意力机制, 通过让多个 Query 头共享同一组 Key/Value 头来减少 KV Cache 的显存占用.

**优势**:

- 显存占用减少 40-60%
- 推理速度提升 20-30%
- 训练性能几乎无损失

**配置**:

```bash
--model.num_heads 32 --model.num_kv_heads 8  # 32个Q头共享8组KV头
```

详见：[Tutorials/预训练](tutorials/01-pretraining.md)

### Q: 什么是 SwiGLU？

SwiGLU 是一种结合 Swish 激活和门控线性单元的激活函数, 相比标准 GELU 能提供更好的性能.

**启用方式**:

```bash
--model.use_glu true
```

详见：[Tutorials/预训练](tutorials/01-pretraining.md)

### Q: 如何选择使用 LayerNorm 还是 RMSNorm？

- **LayerNorm**: 标准选择, 稳定可靠
- **RMSNorm**: 更快的计算速度, 内存占用更少, 效果相当

```bash
--model.norm_impl rms_norm  # 使用 RMSNorm
```

---

## 性能优化

### Q: 如何提升训练速度？

1. **启用混合精度**: 默认已启用
2. **优化数据加载**: 增加 `num_workers`
3. **使用 torch.compile**: 将在未来版本中集成
4. **使用多 GPU**: 见分布式训练问题

### Q: 推理速度慢怎么办？

1. **使用 KVCache**: 见 [Inference Optimization Guide](guides/inference.md)
2. **使用 Top-k 采样**: 减小搜索空间
3. **批处理推理**: 同时处理多个请求
4. **合并 LoRA 权重**: 推理前调用 `merge_lora(model)`

---

## LoRA / QLoRA

### Q: LoRA 和 QLoRA 有什么区别？

| 特性     | LoRA           | QLoRA        |
| -------- | -------------- | ------------ |
| 基础权重 | FP16/FP32      | 4-bit NF4    |
| 内存占用 | ~10% 参数      | ~5% 内存     |
| 推理开销 | 可合并, 无开销 | 需反量化     |
| 适用场景 | 有足够显存     | 显存严重受限 |

### Q: 如何选择 LoRA rank？

- **rank=4-8**: 简单任务, 快速实验
- **rank=16-32**: 复杂任务, 更多容量
- **alpha**: 通常设为 `2 * rank`

详见: [Fine-Tuning Guide](guides/finetuning.md)

---

## 流式训练 (Streaming Pretraining)

### Q: 什么是流式预训练（stream_lm）？

A: 流式预训练使用 StreamingTextDataModule 和 HFStreamTextSource 直接从 HuggingFace 数据集流式读取数据，无需下载到本地。适合大规模预训练场景。

### Q: 如何使用流式预训练？

A: 使用 `uv run llm-train stream_lm --config configs/streaming_c4.yaml`。详情见预训练教程。

### Q: 流式训练的 checkpoint 如何恢复？

A: 流式 checkpoint 除了常规的 model/optimizer/scheduler 状态外，还保存了 `extra_state["stream_data"]` 中的 data cursor，恢复时自动接续上次的 line_index，不会重复读或漏读。

---

## 量化 (Quantization)

### Q: 模型量化支持哪些方法？

A: 目前支持 GPTQ（Frantar 2022）Hessian-aware 4/8-bit PTQ，通过 `llm-quantize gptq` CLI 使用。

### Q: 如何量化一个训练好的模型？

A: 使用 `llm-quantize gptq --model ckpt.pt --output ckpt-int4.pt --calib-data texts.txt --tokenizer gpt2 --bits 4`。

---

## 评估 (Evaluation)

### Q: 如何评估训练好的模型？

A: 项目集成了 lm-evaluation-harness，支持 MMLU、ARC、WikiText 等标准 benchmark。使用 `uv sync --extra eval` 安装依赖后，通过 Python API 运行。

### Q: 如何快速在 MMLU 上评估模型？

A:

```python
from llm.evaluation.harness.lm_eval_lm import LlamaLmEvalLM
from llm.evaluation.harness.adapter import LmEvalAdapter

# ... 加载模型和 tokenizer ...
lm = LlamaLmEvalLM(model, tokenizer, batch_size=8)
raw = LmEvalAdapter().run_preset("mmlu", lm)
print(LmEvalAdapter.summarize(raw))
```

---

## 导出 (Export)

### Q: 模型支持哪些导出格式？

A: 支持 ONNX 和 TorchScript 两种导出格式，通过 EXPORT_REGISTRY 统一调度。

### Q: 如何将模型发布到 HuggingFace Hub？

A:

```python
from llm.compat.hf_publisher import push_to_hub

push_to_hub(model, repo_id="username/my-model")
```

---

## 推理优化

### Q: 什么是 Paged Attention？

A: Paged Attention 将 KV cache 分成固定大小的 block（page），通过 block allocator 管理，减少显存碎片。在 serving 配置中启用 `LLM_SERVING_USE_PAGED_ATTENTION=true`。

### Q: 什么是 Prefix Cache？

A: Prefix Cache 缓存 system prompt 的 KV cache 结果。当多个请求共享相同的 system prompt 时，可以跳过重复计算。在 serving 中启用 `LLM_SERVING_ENABLE_PREFIX_CACHE=true`。

### Q: 什么是 Speculative Decoding？

A: Speculative Decoding 使用一个小 draft 模型快速生成候选 token，大 target 模型在一个 forward pass 中验证并修正。可在高延迟场景下获得 2-3x 吞吐提升。

---

## PEFT 方法

### Q: 框架支持哪些 PEFT 方法？

A: 内置 8 种 PEFT 方法，全部通过统一的 PEFT_REGISTRY 管理：

| 方法             | 类型        | 参数占比 | 适用场景                  |
| ---------------- | ----------- | -------- | ------------------------- |
| LoRA             | 低秩适配    | ~10%     | 通用 PEFT，效果与效率平衡 |
| QLoRA            | 量化 LoRA   | ~5%      | 显存严重受限，大模型      |
| AdaLoRA          | 自适应 LoRA | ~10%     | 自适应秩 + 剪枝           |
| IA³              | 乘性适配    | ~0.01%   | 极轻量，多任务            |
| BitFit           | 偏置微调    | ~0.1%    | 最轻量，快速实验          |
| Adapter          | 瓶颈适配器  | ~5%      | 经典 PEFT                 |
| Pfeiffer Adapter | FFN Adapter | ~2.5%    | Houlsby 变体，参数更少    |
| Prefix Tuning    | 前缀微调    | ~1%      | 指令微调                  |

通过 `training.peft_method` 配置，同一份 YAML 格式切换。

### Q: PEFT adapter 如何保存和加载？

A:

```python
# 保存 adapter（不保存 base 权重）
from llm.core.peft.checkpoint import save_peft

save_peft(model, "adapter.bin", method="lora")

# 加载 adapter
from llm.core.peft.checkpoint import load_peft

load_peft(model, "adapter.bin")
```

训练中自动通过 PEFTAdapterCheckpointCallback 保存，`peft_save_path` 配置路径即可。

### Q: PEFT adapter 如何挂载到推理服务？

A: 通过环境变量：`LLM_SERVING_PEFT_METHOD=lora LLM_SERVING_PEFT_ADAPTER_PATH=./adapter.bin` 即可在 llm-serve 启动时自动加载 adapter。

---

## 开发工具

### Q: 为什么使用 `ty` 而不是 `mypy`？

`ty` 是 Astral 出品的现代类型检查器, 与 Ruff 同系列：

- **速度快**: 比 mypy 快数倍
- **更好的错误信息**: 更清晰的类型错误提示
- **零配置**: 开箱即用

### Q: 为什么使用 `prek` 而不是 `pre-commit`？

`prek` 是更现代的 Git 钩子管理工具：

- **性能更好**: 使用 Rust 编写
- **更简单的配置**: 与项目工具链一致
- **更好的集成**: 原生支持 uv, ruff, ty 等工具

### Q: 如何运行代码质量检查？

```bash
make ruff   # 运行 ruff
make ty     # 运行 ty 类型检查
make test   # 运行测试
```

---

## 其他问题

### Q: 如何贡献代码？

请参考 [Contributing Guide](https://github.com/pplmx/llm/blob/main/CONTRIBUTING.md) 了解详细流程.

### Q: 在哪里报告 Bug？

请在 [GitHub Issues](https://github.com/pplmx/llm/issues) 提交 bug 报告, 使用 bug report 模板.

### Q: 如何获取帮助？

1. 查看本 FAQ 和其他文档
2. 查看 [Troubleshooting Guide](troubleshooting.md)
3. 在 GitHub Discussions 提问
4. 提交 Issue(如果是 bug)

---

**找不到答案？** 欢迎在 [GitHub Discussions](https://github.com/pplmx/llm/discussions) 提问！
