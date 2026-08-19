# LLM 框架文档

<div class="hero" markdown>

一个模块化、可扩展的 PyTorch 大语言模型框架：**流式预训练 → PEFT 微调 →
量化压缩 → 评估 → OpenAI 兼容推理服务 → 多格式导出**，覆盖 LLM 的完整生命周期。

[🚀 快速开始](getting-started.md){ .md-button .md-button--primary }
[📖 教程](tutorials/01-pretraining.md){ .md-button }
[🔧 功能指南](guides/index.md){ .md-button }
[🔌 API Reference](api/index.md){ .md-button }

</div>

<div class="feature-grid" markdown>

<div class="feature" markdown>
**⚡ 流式预训练**

`stream_lm` 主路径 + C4 / Pile / RedPajama 数据预设，支持 dedup 与精确续训。
</div>

<div class="feature" markdown>
**🧩 8 种 PEFT**

LoRA / QLoRA / AdaLoRA / IA³ / BitFit / Adapter / Pfeiffer / Prefix Tuning，
统一 `PEFT_REGISTRY` 管理。
</div>

<div class="feature" markdown>
**🚀 推理服务**

OpenAI 兼容 API、KV cache、Paged Attention、Prefix Cache 与连续批处理。
</div>

<div class="feature" markdown>
**🔢 量化与导出**

GPTQ / AWQ / SmoothQuant / 混合精度；ONNX / TorchScript / GGUF 一键导出。
</div>

<div class="feature" markdown>
**📊 离线评估**

lm-evaluation-harness 集成，MMLU / ARC / WikiText 预设即用。
</div>

<div class="feature" markdown>
**🔌 插件化架构**

Registry + entry points，模型、数据源、后端、PEFT 均可扩展。
</div>

</div>

---

## 🎯 按需求选择入口

### 🆕 刚上手

- [快速开始](getting-started.md) — 5 分钟跑通第一个训练
- [预训练教程](tutorials/01-pretraining.md) — 流式大规模预训练全流程
- [微调教程](tutorials/02-finetuning.md) — SFT / DPO + LoRA/QLoRA
- [推理教程](tutorials/03-inference.md) — `llm-serve` 部署与 OpenAI SDK 集成

### 🚀 训练与数据

| 指南                                         | 说明                                |
| -------------------------------------------- | ----------------------------------- |
| [数据管线](guides/data.md)                   | 数据源、数据集预设、`DataConfig`    |
| [流式预训练](guides/streaming.md)            | `stream_lm` 流式 pipeline           |
| [Checkpoint](guides/checkpoints.md)          | v2 三件套、保存/恢复/迁移           |
| [分布式训练](guides/distributed.md)          | DDP / FSDP / Tensor 并行 (TP+DP 2D) |
| [PEFT 微调](guides/finetuning.md)            | 8 种参数高效微调方法                |
| [训练流程开发](development/training-flow.md) | 引擎、回调、调度器扩展              |

### ⚡ 部署与压缩

| 指南                               | 说明                                    |
| ---------------------------------- | --------------------------------------- |
| [推理优化](guides/inference.md)    | KV cache、Paged Attention、prefix cache |
| [模型量化](guides/quantization.md) | GPTQ / AWQ / SmoothQuant / PTQ          |
| [模型导出](guides/export.md)       | ONNX / TorchScript / GGUF               |
| [模型评估](guides/evaluation.md)   | lm-evaluation-harness 集成              |

### 📚 参考

| 文档                                           | 说明                          |
| ---------------------------------------------- | ----------------------------- |
| [系统架构](reference/architecture.md)          | 分层设计、Registry 机制       |
| [CLI 命令](reference/cli.md)                   | `llm-train` / `llm-serve` 等  |
| [API Reference](api/index.md)                  | 按子包组织的自动生成 API 文档 |
| [架构决策 (ADR)](adr/README.md)                | 已锁定的架构决策与理由        |
| [开发指南](development/README.md)              | 组件、扩展、DDP 深入          |
| [FAQ](faq.md) / [故障排除](troubleshooting.md) | 常见问题与排查                |

---

## 📁 文档结构

```text
docs/
├── README.md                # 🏠 本页（概览）
├── getting-started.md       # 🚀 快速开始
├── tutorials/               # 📖 教程（预训练 / 微调 / 推理）
├── guides/                  # 🔧 功能指南（index + 数据 / 量化 / 导出…）
├── reference/               # 📚 架构与 CLI 参考
├── api/                     # 🔌 自动生成的 API Reference
├── adr/                     # 📝 架构决策记录
├── development/             # 💻 开发者文档
├── tags.md                  # 🏷️ 标签索引
└── stylesheets/extra.css    # 🎨 站点自定义样式
```

---

## ⚡ 快速命令

<div class="quick-commands" markdown>

```bash
# 安装依赖与 pre-commit hooks
make init

# 流式预训练（本地冒烟，CPU 可跑）
uv run llm-train --task stream_lm --config-path configs/streaming_local_demo.yaml

# 启动 OpenAI 兼容推理服务
uv run llm-serve

# GPTQ 量化
uv run llm-quantize gptq --model ckpt.pt --output ckpt-int4.pt \
    --calib-data texts.txt --tokenizer gpt2 --bits 4

# 质量门槛
make ruff && make ty && make test-fast
```

</div>

---

## 🤝 贡献

欢迎贡献！请阅读 [CONTRIBUTING.md](https://github.com/pplmx/llm/blob/main/CONTRIBUTING.md)
了解参与方式，并在 [GitHub Issues](https://github.com/pplmx/llm/issues) 反馈问题。
