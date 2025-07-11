# Gemini Workspace

This file provides context to the Gemini agent about the project.

## Project Overview

This project is a Python application named "llm". It uses `uv` for package management and `hatchling` for building.

## Commit Message Guidelines

This project adheres to the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification for commit messages. Please use the following types:

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc.)
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `build`: Changes that affect the build system or external dependencies (example scopes: pip, docker, npm)
- `ci`: Changes to our CI configuration files and scripts (example scopes: Travis, Circle, BrowserStack, SauceLabs)
- `chore`: Other changes that don't modify src or test files
- `revert`: Reverts a previous commit

**Note:** The `description` (subject line) should generally start with a lowercase letter.

## Commit Workflow

To ensure consistent and properly formatted commit messages, please follow this workflow:

1. **Create a temporary file:** Write the full commit message, including the subject and body, to a temporary file named `commit_message.txt` in the project root.
2. **Commit from the file:** Use the command `git commit -F commit_message.txt` to create the commit.
3. **Clean up:** After the commit is successfully created, delete the `commit_message.txt` file.

## Commands

The following commands are available in the `Makefile`:

- `make init`: Initializes the virtual environment and installs pre-commit hooks.
- `make sync`: Syncs the project dependencies.
- `make build`: Builds the project wheel.
- `make test`: Runs tests using `pytest`.
- `make allure`: Serves the Allure test report.
- `make ruff`: Formats and lints the code with `ruff`.
- `make type`: Performs static type checking with `mypy`.
- `make image`: Builds a Docker image for the application.
- `make compose-up`: Starts the application using Docker Compose.
- `make compose-down`: Stops the application using Docker Compose.
- `make clean`: Removes build artifacts and stops Docker containers.
- `make help`: Shows a help message with all available commands.

## Tooling

- **Linting and Formatting:** `ruff` is used for both linting and formatting. The configuration is in `ruff.toml`.
- **Type Checking:** `mypy` is used for static type checking. The configuration is in `pyproject.toml`.
- **Testing:** `pytest` is used for running tests. The configuration is in `pyproject.toml`.
- **Package Management:** `uv` is used for managing dependencies.
- **Building:** `hatchling` is used for building the project.

## Dependencies

The main dependencies are listed in `pyproject.toml` and include:

- `pytest`
- `pytest-cov`
- `allure-pytest`
- `torch`
- `matplotlib`
- `seaborn`
- `pillow`

## 文档结构规范

项目的文档组织在 `docs/` 目录下，遵循标准的 GitHub 项目实践。关键规范包括：

-   **根目录的 `CONTRIBUTING.md`**: 提供高层次的贡献指南。
-   **`docs/` 目录作为主要文档中心**:
    -   **根目录的 `README.md` 是所有文档的主要入口**，它直接链接到 `docs/` 目录下的各个顶级文档文件。
    -   大多数文档文件名使用小写（例如 `development.md`, `tutorial-cpu-llm.md`），`README.md` 是例外。
    -   相关文档组织在特定子目录中（例如 `docs/training/` 用于所有训练框架文档）。
    -   子目录可以包含自己的 `README.md` 作为入口。
    -   项目级别的 `docs/troubleshooting.md` 用于通用问题排查。
