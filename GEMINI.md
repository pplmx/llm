# Gemini Workspace

This file provides context to the Gemini agent about the project.

## Project Overview

This project is a Python application named "llm". It uses `uv` for package management and `hatchling` for building.

## Commit Message Guidelines

This project adheres to the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification for commit messages. Please use the following types:

*   `feat`: A new feature
*   `fix`: A bug fix
*   `docs`: Documentation only changes
*   `style`: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc.)
*   `refactor`: A code change that neither fixes a bug nor adds a feature
*   `perf`: A code change that improves performance
*   `test`: Adding missing tests or correcting existing tests
*   `build`: Changes that affect the build system or external dependencies (example scopes: pip, docker, npm)
*   `ci`: Changes to our CI configuration files and scripts (example scopes: Travis, Circle, BrowserStack, SauceLabs)
*   `chore`: Other changes that don't modify src or test files
*   `revert`: Reverts a previous commit

**Note:** The `description` (subject line) should generally start with a lowercase letter.

## Commands

The following commands are available in the `Makefile`:

*   `make init`: Initializes the virtual environment and installs pre-commit hooks.
*   `make sync`: Syncs the project dependencies.
*   `make build`: Builds the project wheel.
*   `make test`: Runs tests using `pytest`.
*   `make allure`: Serves the Allure test report.
*   `make ruff`: Formats and lints the code with `ruff`.
*   `make type`: Performs static type checking with `mypy`.
*   `make image`: Builds a Docker image for the application.
*   `make compose-up`: Starts the application using Docker Compose.
*   `make compose-down`: Stops the application using Docker Compose.
*   `make clean`: Removes build artifacts and stops Docker containers.
*   `make help`: Shows a help message with all available commands.

## Tooling

*   **Linting and Formatting:** `ruff` is used for both linting and formatting. The configuration is in `ruff.toml`.
*   **Type Checking:** `mypy` is used for static type checking. The configuration is in `pyproject.toml`.
*   **Testing:** `pytest` is used for running tests. The configuration is in `pyproject.toml`.
*   **Package Management:** `uv` is used for managing dependencies.
*   **Building:** `hatchling` is used for building the project.

## Dependencies

The main dependencies are listed in `pyproject.toml` and include:

*   `pytest`
*   `pytest-cov`
*   `allure-pytest`
*   `torch`
*   `matplotlib`
*   `seaborn`
*   `pillow`
