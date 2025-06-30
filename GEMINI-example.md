# Gemini Workspace: [Project Name]

This file provides context to the Gemini agent about the project. Please fill in the bracketed placeholders (`[...]`) with your project's specific details.

## 1. Project Overview

- **Project Name:** `[Project Name]`
- **Description:** `[A brief, one-sentence description of the project's purpose.]`
- **Primary Language/Framework:** `[e.g., Python/FastAPI, TypeScript/React, Go]`
- **Package/Dependency Manager:** `[e.g., uv, pip, npm, yarn, go mod]`
- **Build Tool:** `[e.g., hatchling, webpack, esbuild, Make]`

## 2. Key Commands

List the most common commands needed for development. This helps the agent perform routine tasks like testing, building, and linting without assistance.

- `[make init]`: `[e.g., Initializes the development environment and installs dependencies.]`
- `[make test]`: `[e.g., Runs the test suite using pytest.]`
- `[make lint]`: `[e.g., Formats and lints the codebase using ruff.]`
- `[make build]`: `[e.g., Compiles the project or builds the application artifact.]`
- `[make run]`: `[e.g., Starts the development server.]`
- `[make ...]`

## 3. Tooling & Configuration

Describe the primary development tools and where their configurations are located.

- **Linting/Formatting:** `[e.g., Ruff, ESLint, Prettier]. Configuration is in [e.g., ruff.toml, .eslintrc.js].`
- **Type Checking:** `[e.g., MyPy, TypeScript]. Configuration is in [e.g., pyproject.toml, tsconfig.json].`
- **Testing:** `[e.g., Pytest, Jest]. Configuration is in [e.g., pyproject.toml, jest.config.js].`
- **Dependency Management:** `[e.g., uv, npm]. Dependencies are listed in [e.g., pyproject.toml, package.json].`

## 4. Commit Message Guidelines

Specify the commit message convention used in the project.

- **Convention:** `[e.g., Conventional Commits, or a custom format.]`
- **Example Types:**
  - `feat`: A new feature
  - `fix`: A bug fix
  - `docs`: Documentation only changes
  - `style`: Code style changes (formatting, etc.)
  - `refactor`: A code change that neither fixes a bug nor adds a feature
  - `test`: Adding or correcting tests
  - `chore`: Routine tasks, maintenance

## 5. Code Style & Conventions

Outline any important coding styles or architectural patterns.

- **Formatting:** `[e.g., Adheres to Black, with a line length of 88.]`
- **Naming Conventions:** `[e.g., Functions are snake_case, classes are PascalCase.]`
- **Architectural Patterns:** `[e.g., Follows a layered architecture (controller, service, repository).]`
- **API Style:** `[e.g., RESTful, with endpoints documented via OpenAPI specs in docs/api.yml.]`

## 6. Important Directory Structure

Highlight key directories and their purpose.

- `src/`: Main source code for the application.
- `tests/`: Contains all unit, integration, and end-to-end tests.
- `docs/`: Project documentation.
- `scripts/`: Automation or utility scripts.
- `[...]/`

## 7. Agent Preferences & Workflow

This section is for documenting established workflows with the agent.

- **Commit Workflow:** To ensure consistent commit messages, we first write the message to a temporary `commit_message.txt` file, then use `git commit -F commit_message.txt`, and finally delete the file.
- **Language Preference:** `[e.g., Please communicate in English.]` (Note: This is better saved to the agent's memory, but can be noted here for project-wide consistency if needed).
