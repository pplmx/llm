# Development Guide for llm

Welcome to the development guide for `llm`!
This document will walk you through setting up your development environment, running tests, building the project, and maintaining code quality.

## Table of Contents

- [Setting Up the Development Environment](#setting-up-the-development-environment)
    - [Prerequisites](#prerequisites)
    - [Installation Steps](#installation-steps)
- [Running Tests](#running-tests)
- [Building the Project](#building-the-project)
- [Code Style and Linting](#code-style-and-linting)

## Setting Up the Development Environment

### Prerequisites

Before you start, make sure you have the following installed on your system:

- **Python 3.13+**: Ensure you have the correct version of Python. You can check your Python version with:

    ```bash
    python --version
    ```

- **`uv` tool**: This tool helps manage your Python environment.

    - **macOS and Linux**:

        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ```

    - **Windows**:

        ```bash
        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```

### Installation Steps

1.  **Clone the Repository**: Start by cloning the project repository to your local machine and navigate to the project directory:
    ```bash
    git clone https://github.com/pplmx/llm.git
    cd llm
    ```

2.  **Synchronize Dependencies**: Use `uv` to set up the virtual environment and install all necessary dependencies (including development tools) as defined in `pyproject.toml`. `uv` will automatically create or use a virtual environment in `.venv` at the project root.
    ```bash
    uv sync
    ```

3.  **Set Up Pre-commit Hooks**: This project uses pre-commit hooks to ensure code quality and consistency before commits are made. Install the hooks with:
    ```bash
    uvx pre-commit install --hook-type commit-msg --hook-type pre-push
    ```
    This step is important to run after cloning and setting up the environment.

## Running Tests

Tests are managed and run using `pytest`. Ensure your dependencies are synchronized with `uv sync` before running tests.

-   **Run all tests**:
    ```bash
    uv run pytest
    ```
    This command discovers and executes all tests in the `tests/` directory.

[Consider adding specific details on the structure of tests, testing strategy, or how to add new tests.]

## Building the Project

To build the project and create distributable packages (e.g., `.whl` and source distribution), use `hatchling` via `uvx`:

```bash
uvx hatch build
```

This command, as configured by `pyproject.toml` (using `hatchling.build` as the build backend), will generate the distributable files in the `dist/` directory.

## Code Style and Linting

Maintaining consistent code style and quality is essential. We use `Ruff` for both formatting and linting. All commands should be run from the project root.

-   **Format code (apply changes)**:
    ```bash
    uvx ruff format .
    ```
    This command automatically reformats your code to match the project's style.

-   **Check formatting (without applying changes)**:
    ```bash
    uvx ruff format --check .
    ```
    This command reports any files that don't adhere to the style guide, without modifying them. Useful for CI checks.

-   **Lint code (check for errors and style issues)**:
    ```bash
    uvx ruff check .
    ```
    This command analyzes your code for potential errors, bugs, and style violations.

-   **Lint code and apply auto-fixes (for safe fixes)**:
    ```bash
    uvx ruff check . --fix
    ```
    This command attempts to automatically fix any safe linting issues found.

-   **Lint code and apply more aggressive auto-fixes (including potentially unsafe ones)**:
    ```bash
    uvx ruff check . --fix --unsafe-fixes
    ```
    Use this with caution, as it might apply changes that alter semantics in rare cases.

---

By following this guide, you'll be well on your way to contributing to `llm`. Thank you for your efforts in maintaining and improving this project!
