# Agent 指南

此文件为参与本项目的 AI Agent 提供快速上下文引导.

## 快速索引

| 文件 | 用途 |
| ------ | ------ |
| [GEMINI.md](GEMINI.md) | Gemini 专属完整规范 |
| [ROADMAP.md](ROADMAP.md) | 项目状态与开发路线图 |
| [docs/](docs/) | 技术文档 (架构、训练、教程) |
| [pyproject.toml](pyproject.toml) | 项目配置与依赖 |

## 核心规范

### 代码质量

- 任何改动后必须运行 `make test`
- 使用 `make ruff` 进行格式化和 lint
- 始终使用 `pathlib.Path` 处理文件路径

### 提交规范

- 遵循 [Conventional Commits](https://www.conventionalcommits.org/)
- 使用 `commit_message.txt` 工作流 (创建 → `git commit -F` → 删除)
- 保持小而聚焦的 commit

### 验证流程

每次改动后执行: `make test` → `make ruff` → `make test`

## 用户偏好

- 使用**中文**交流
- 代码注释使用**英文/半角标点**
- 不使用 mock, 直接调用真实代码测试
- 遇到 bug 时考虑是否需要添加测试覆盖
