# Gemini 工作区

此文件为 Gemini Agent 提供项目上下文信息.

## 项目概览

本项目是一个名为 "llm" 的 Python 应用程序, 旨在构建一个现代、高效的 Decoder-only Transformer 架构.
使用 `uv` 进行包管理, 项目配置详见 `pyproject.toml`.

### 核心架构特性

- **Decoder-only Transformer**: 基础架构.
- **SwiGLU 激活函数**: 提升 MLP 层的表现.
- **Grouped Query Attention (GQA)**: 兼顾性能与显存效率.
- **统一 QKV 投影**: 优化内存布局与计算吞吐.
- **混合精度支持**: 包含自动 BF16/FP16 检测.
- **BPE 分词器**: 高效的子词词元化.

## 提交信息规范

本项目遵循 [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) 规范来编写提交信息. 请使用以下类型:

- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 仅文档更改
- `style`: 不影响代码含义的更改(例如: 空白、格式、缺少分号等)
- `refactor`: 既不修复 Bug 也不添加功能的代码更改
- `perf`: 提高性能的代码更改
- `test`: 添加缺失的测试或更正现有测试
- `build`: 影响构建系统或外部依赖的更改(示例范围: pip, docker, npm)
- `ci`: 更改 CI 配置文件和脚本(示例范围: Travis, Circle, BrowserStack, SauceLabs)
- `chore`: 不修改 `src` 或 `test` 文件的其他更改
- `revert`: 撤销之前的提交

**注意**: `description`(主题行)通常应以小写字母开头.

## 提交工作流

为确保提交信息的一致性和正确格式, 请遵循以下工作流:

1. **创建临时文件**: 将完整的提交信息(包括主题和正文)写入项目根目录下的临时文件 `commit_message.txt`.
2. **从文件提交**: 使用命令 `git commit -F commit_message.txt` 创建提交.
3. **清理**: 成功创建提交后, 删除 `commit_message.txt` 文件.

## 可用命令

运行 `make help` 查看所有可用的 Makefile 命令.

## 工具链

- **Linting 和格式化**: `ruff` 用于 linting 和格式化. 配置位于 `ruff.toml`. 遵循 `pathlib` 最佳实践.
- **类型检查**: `mypy` 用于静态类型检查. 配置位于 `pyproject.toml`.
- **测试**: `pytest` 用于运行测试. 配置位于 `pyproject.toml`.
- **包管理**: `uv` 用于管理依赖.
- **构建**: 详见 `pyproject.toml` 的 build-system 配置.

## 依赖项

主要依赖项详见 `pyproject.toml`.

## 文档结构规范

项目的文档组织在 `docs/` 目录下, 遵循标准的 GitHub 项目实践. 关键规范包括:

- **根目录的 `CONTRIBUTING.md`**: 提供高层次的贡献指南.
- **`docs/` 目录作为主要文档中心**:
    - **根目录的 `README.md` 是所有文档的主要入口**, 它直接链接到 `docs/` 目录下的各个顶级文档文件.
    - 大多数文档文件名使用小写, **多单词文件名使用连字符 `-` 分隔**(例如 `development.md`, `tutorial-cpu-llm.md`,
      `guide-extending.md`), `README.md` 是例外.
    - 相关文档组织在特定子目录中(例如 `docs/training/` 用于所有训练框架文档).
    - 子目录可以包含自己的 `README.md` 作为入口.
    - 项目级别的 `docs/troubleshooting.md` 用于通用问题排查.

## Comments policy

Only write high-value comments if at all. Avoid talking to the user through comments.

## 用户偏好与开发规范

- 用户偏好我使用中文进行回复.
- 用户极其看重代码美感和现代化(如：使用 `pathlib` 替代 `os.path`, 避免冗余注释).
- 标点符号规范：在代码注释和文档中, 偏好使用英文/半角标点符号(如 `,` `(` `)`), 避免使用全角标点.
- 测试偏好：在 Python 项目中测试时, 偏好不使用 mock, 直接调用代码, 主要验证功能性, 不关注覆盖率, 只使用 pytest.
- 架构一致性：任何对 MHA 或 MLP 的修改必须考虑 GQA 和 SwiGLU 的兼容性.
- **验证流程**：每次 Plan 实施结束后，必须执行 `make test -> make ruff -> make test` 验证流程，确保代码质量和功能正确性.
- **Commit 粒度**: 保持小而聚焦的 commit, 除非功能无法合理拆分.
- **错误驱动测试**: 遇到 bug 或边界情况时, 考虑是否需要添加测试覆盖.

---

## 开发路线图

项目的详细开发路线图请参见 [ROADMAP.md](ROADMAP.md).
