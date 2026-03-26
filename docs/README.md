# LLM 项目文档

欢迎来到 LLM 项目文档。根据你的需求选择合适的入口：

---

## 🎯 我要做什么？

### 🚀 快速开始 (5分钟)

还不知道这是什么？先看这里：

- [快速开始](getting-started.md) - 5分钟运行第一个训练

### 📖 学习教程

一步步跟着做：

| 教程                                  | 说明            | 预计时间 |
| ------------------------------------- | --------------- | -------- |
| [预训练](tutorials/01-pretraining.md) | 从零训练模型    | 15分钟   |
| [微调](tutorials/02-finetuning.md)    | LoRA/QLoRA 微调 | 20分钟   |
| [推理](tutorials/03-inference.md)     | 模型推理和部署  | 10分钟   |

### 🔧 功能指南

深入了解特定功能：

| 指南                                 | 说明            |
| ------------------------------------ | --------------- |
| [Checkpoints](guides/checkpoints.md) | 模型保存与恢复  |
| [分布式训练](guides/distributed.md)  | 多 GPU 训练     |
| [Fine-Tuning](guides/finetuning.md)  | LoRA/QLoRA 详解 |
| [Inference](guides/inference.md)     | 推理优化        |

### 📚 参考文档

快速查找：

| 文档                                      | 说明       |
| ----------------------------------------- | ---------- |
| [CLI 命令](reference/cli.md)              | 命令行参数 |
| [Architecture](reference/architecture.md) | 系统架构   |
| [Development](development/README.md)      | 开发者文档 |

---

## 📁 文档结构

```text
docs/
├── getting-started.md          # 🚀 快速开始
├── tutorials/                # 📖 入门教程
│   ├── 01-pretraining.md   # 预训练
│   ├── 02-finetuning.md   # 微调
│   └── 03-inference.md   # 推理
├── guides/                  # 🔧 功能指南
│   ├── checkpoints.md
│   ├── distributed.md
│   ├── finetuning.md
│   └── inference.md
├── reference/               # 📚 参考
│   ├── architecture.md
│   └── cli.md
└── development/            # 💻 开发者文档 (高级)
```

---

## 快速命令

```bash
# 安装
make init

# 训练
uv run scripts/train_simple_decoder.py --file-path data.txt

# 测试
make test

# Lint
make ruff
```

---

## 需要帮助？

- [FAQ](faq.md) - 常见问题
- [Troubleshooting](troubleshooting.md) - 故障排除
- [GitHub Issues](https://github.com/your-repo/llm/issues) - 报告问题

---

## 贡献

欢迎贡献！请阅读 [CONTRIBUTING.md](../CONTRIBUTING.md) 了解如何参与。
