# Agent 指南

此文件为参与本项目的 AI Agent 提供快速上下文引导. 详细规则请参考 `GEMINI.md`.

## 核心任务

1. **架构现代化**: 维护和优化已实现的 SwiGLU、GQA 和统一 QKV 投影.
2. **代码质量**: 严格执行 `ruff` 检查. 确保所有修改遵循 `pathlib` 最佳实践.
3. **测试驱动**: 任何改动后必须运行 `make test`. 目前共有 299 个测试用例.

## 关键技术栈

- **Core**: PyTorch (Llama 风格架构)
- **Tooling**: `uv`, `ruff`, `pytest`
- **Normalization**: 支持 RMSNorm 和 LayerNorm
- **Tokenization**: `tokenizers` 驱动的 BPETokenizer

## Agent 行为准则

- **语言**: 优先使用中文交流.
- **格式**: 代码注释和文档中使用英文/半角标点.
- **提交**: 遵循 Conventional Commits, 并使用 `commit_message.txt` 工作流.
- **路径**: 始终使用 `pathlib.Path` 处理文件路径.
- **验证**: 每次大改动后依次执行 `make test` -> `make ruff` -> `make test`.

## 当前状态 (2026-01-05)

- [x] SwiGLU 实现与集成
- [x] GQA 与统一 QKV 投影
- [x] BPETokenizer 集成
- [x] BF16 自动检测
- [x] 代码质量全面清理 (Ruff clean)
- [x] FastAPI 推理服务化 (流式 + 批处理)
- [ ] 请求队列与并发控制
- [ ] Flash Attention 2 集成
