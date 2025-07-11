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
- [Type Checking](#type-checking)
- [Docker Development](#docker-development)

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

- **`make` tool**: A build automation tool (typically pre-installed on Linux/macOS, available via Chocolatey/Scoop on Windows).

### Installation Steps

1.  **Clone the Repository**: Start by cloning the project repository to your local machine and navigate to the project directory:
    ```bash
    git clone https://github.com/pplmx/llm.git
    cd llm
    ```

2.  **Initialize Project**: Run the `make init` command to set up the virtual environment, install all necessary dependencies, and install pre-commit hooks.
    ```bash
    make init
    ```
    This command internally runs `uv sync` and `uvx pre-commit install --hook-type commit-msg --hook-type pre-push`.

## Running Tests

Tests are managed and run using `pytest`.

-   **Run all tests**:
    ```bash
    make test
    ```
    This command discovers and executes all tests in the `tests/` directory, generates code coverage reports (HTML, LCOV, XML) in the `htmlcov/` directory, and Allure test results in `allure-results/`. You can view the Allure report by running `make allure`.

## Building the Project

To build the project and create distributable packages (e.g., `.whl` and source distribution), use the `make build` command:

```bash
make build
```

This command, as configured by `pyproject.toml` (using `hatchling.build` as the build backend), will generate the distributable files in the `dist/` directory.

## Code Style and Linting

Maintaining consistent code style and quality is essential. We use `Ruff` for both formatting and linting.

-   **Run Code Style & Linting Checks**:
    ```bash
    make ruff
    ```
    This command will format your code and check for linting issues. For more granular control, you can use the `uvx ruff` commands directly as follows:

    -   **Format code (apply changes)**:
        ```bash
        uvx ruff format .
        ```
    -   **Check formatting (without applying changes)**:
        ```bash
        uvx ruff format --check .
        ```
    -   **Lint code (check for errors and style issues)**:
        ```bash
        uvx ruff check .
        ```
    -   **Lint code and apply auto-fixes (for safe fixes)**:
        ```bash
        uvx ruff check . --fix
        ```
    -   **Lint code and apply more aggressive auto-fixes (including potentially unsafe ones)**:
        ```bash
        uvx ruff check . --fix --unsafe-fixes
        ```

## Type Checking

This project uses `mypy` for static type checking to ensure code correctness and maintainability.

-   **Run type checks**:
    ```bash
    make type
    ```
    This command will run `mypy` against the codebase based on the configuration in `pyproject.toml`.

## Docker Development

The project provides `Makefile` commands for Docker-related tasks.

-   **Build Docker Image**:
    ```bash
    make image
    ```
    Builds the application's Docker image.

-   **Start Application with Docker Compose**:
    ```bash
    make compose-up
    ```
    Starts the application using Docker Compose.

-   **Stop Application with Docker Compose**:
    ```bash
    make compose-down
    ```
    Stops the application started with Docker Compose.

-   **Clean Project**:
    ```bash
    make clean
    ```
    Removes build artifacts and stops Docker containers.

---

By following this guide, you'll be well on your way to contributing to `llm`. Thank you for your efforts in maintaining and improving this project!
