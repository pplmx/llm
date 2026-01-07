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

详见：[Development Guide](development.md)

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
llm-train --task lm --epochs 10 --batch-size 32
```

### Q: 如何启用分布式训练？

使用 `torchrun` 启动多 GPU 训练：

```bash
torchrun --nproc_per_node=4 src/llm/training/train.py --task lm
```

详见：[DDP Deep Dive](training/deep-dive-ddp.md)

### Q: 内存不足 (OOM) 怎么办？

尝试以下方法：

1. **减小 batch size**: `--batch-size 16`
2. **启用混合精度**: 默认已启用 AMP
3. **减小模型大小**: `--model.hidden_size 512`
4. **使用 Gradient Checkpointing**: 将在未来版本中支持

详见：[Troubleshooting Guide](troubleshooting.md#内存不足)

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

详见：[Tutorial](tutorial-cpu-llm.md#gqa-说明)

### Q: 什么是 SwiGLU？

SwiGLU 是一种结合 Swish 激活和门控线性单元的激活函数, 相比标准 GELU 能提供更好的性能.

**启用方式**:

```bash
--model.use_glu true
```

详见：[Tutorial](tutorial-cpu-llm.md#swiglu-说明)

### Q: 如何选择使用 LayerNorm 还是 RMSNorm？

- **LayerNorm**: 标准选择, 稳定可靠
- **RMSNorm**: 更快的计算速度, 内存占用更少, 效果相当

```bash
--model.norm_type RMSNorm  # 使用 RMSNorm
```

---

## 性能优化

### Q: 如何提升训练速度？

1. **启用混合精度**: 默认已启用
2. **优化数据加载**: 增加 `num_workers`
3. **使用 torch.compile**: 将在未来版本中集成
4. **使用多 GPU**: 见分布式训练问题

### Q: 推理速度慢怎么办？

1. **确保启用 KV Cache**: 默认在 `generate` 函数中启用
2. **使用 Top-k 采样**: 减小搜索空间
3. **批处理推理**: 同时处理多个请求
4. **等待 Flash Attention**: 将在未来版本中集成

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

请参考 [Contributing Guide](../CONTRIBUTING.md) 了解详细流程.

### Q: 在哪里报告 Bug？

请在 [GitHub Issues](https://github.com/pplmx/llm/issues) 提交 bug 报告, 使用 bug report 模板.

### Q: 如何获取帮助？

1. 查看本 FAQ 和其他文档
2. 查看 [Troubleshooting Guide](troubleshooting.md)
3. 在 GitHub Discussions 提问
4. 提交 Issue(如果是 bug)

---

**找不到答案？** 欢迎在 [GitHub Discussions](https://github.com/pplmx/llm/discussions) 提问！
