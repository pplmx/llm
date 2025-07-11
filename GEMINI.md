# Gemini 工作区

此文件为 Gemini Agent 提供项目上下文信息。

## 项目概览

本项目是一个名为 "llm" 的 Python 应用程序。它使用 `uv` 进行包管理，并使用 `hatchling` 进行构建。

## 提交信息规范

本项目遵循 [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) 规范来编写提交信息。请使用以下类型：

- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 仅文档更改
- `style`: 不影响代码含义的更改（例如：空白、格式、缺少分号等）
- `refactor`: 既不修复 Bug 也不添加功能的代码更改
- `perf`: 提高性能的代码更改
- `test`: 添加缺失的测试或更正现有测试
- `build`: 影响构建系统或外部依赖的更改（示例范围：pip, docker, npm）
- `ci`: 更改 CI 配置文件和脚本（示例范围：Travis, Circle, BrowserStack, SauceLabs）
- `chore`: 不修改 `src` 或 `test` 文件的其他更改
- `revert`: 撤销之前的提交

**注意**：`description`（主题行）通常应以小写字母开头。

## 提交工作流

为确保提交信息的一致性和正确格式，请遵循以下工作流：

1.  **创建临时文件**：将完整的提交信息（包括主题和正文）写入项目根目录下的临时文件 `commit_message.txt`。
2.  **从文件提交**：使用命令 `git commit -F commit_message.txt` 创建提交。
3.  **清理**：成功创建提交后，删除 `commit_message.txt` 文件。

## 可用命令

`Makefile` 中提供了以下命令：

- `make init`: 初始化虚拟环境并安装 pre-commit 钩子。
- `make sync`: 同步项目依赖。
- `make build`: 构建项目 wheel 包。
- `make test`: 使用 `pytest` 运行测试。
- `make allure`: 启动 Allure 测试报告服务。
- `make ruff`: 使用 `ruff` 格式化和 lint 代码。
- `make type`: 使用 `mypy` 执行静态类型检查。
- `make image`: 构建应用程序的 Docker 镜像。
- `make compose-up`: 使用 Docker Compose 启动应用程序。
- `make compose-down`: 使用 Docker Compose 停止应用程序。
- `make clean`: 删除构建产物并停止 Docker 容器。
- `make help`: 显示所有可用命令的帮助信息。

## 工具链

-   **Linting 和格式化**：`ruff` 用于 linting 和格式化。配置位于 `ruff.toml`。
-   **类型检查**：`mypy` 用于静态类型检查。配置位于 `pyproject.toml`。
-   **测试**：`pytest` 用于运行测试。配置位于 `pyproject.toml`。
-   **包管理**：`uv` 用于管理依赖。
-   **构建**：`hatchling` 用于构建项目。

## 依赖项

主要依赖项列在 `pyproject.toml` 中，包括：

- `pytest`
- `pytest-cov`
- `allure-pytest`
- `torch`
- `matplotlib`
- `seaborn`
- `pillow`

## 文档结构规范

项目的文档组织在 `docs/` 目录下，遵循标准的 GitHub 项目实践。关键规范包括：

-   **根目录的 `CONTRIBUTING.md`**：提供高层次的贡献指南。
-   **`docs/` 目录作为主要文档中心**：
    -   **根目录的 `README.md` 是所有文档的主要入口**，它直接链接到 `docs/` 目录下的各个顶级文档文件。
    -   大多数文档文件名使用小写（例如 `development.md`, `tutorial-cpu-llm.md`），`README.md` 是例外。
    -   相关文档组织在特定子目录中（例如 `docs/training/` 用于所有训练框架文档）。
    -   子目录可以包含自己的 `README.md` 作为入口。
    -   项目级别的 `docs/troubleshooting.md` 用于通用问题排查。

## 与 Gemini Agent 协作

为了更高效地与 Gemini Agent 协作，请遵循以下指南：

-   **清晰明确地提问**：请用简洁明了的语言描述您的需求或问题。
-   **提供充足的上下文**：在提问时，尽可能提供相关的代码片段、文件路径、错误信息或您认为有用的任何其他背景信息。
-   **逐步进行**：对于复杂的任务，我们可以分步骤完成。您可以先提出一个高层次的目标，然后根据我的反馈逐步细化。
-   **验证和反馈**：在我完成任务或提供建议后，请您验证结果并提供反馈。这有助于我更好地理解您的期望并持续改进。
-   **文件操作**：当需要我修改文件时，请明确指出文件路径和您希望进行的具体更改。例如：“请在 `src/main.py` 中添加一个名为 `my_function` 的函数。”
-   **运行命令**：如果您希望我运行特定的 shell 命令，请直接提供命令。我会在执行前解释其作用（如果涉及文件系统修改）。

通过遵循这些指南，我们可以更有效地合作，共同完成项目任务。
