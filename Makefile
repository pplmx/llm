.PHONY: help build test ruff image clean lint fmt sync-dev
.DEFAULT_GOAL := help

APP_NAME := llm

# Init the venv
init: sync
	@uvx prek install --hook-type commit-msg --hook-type pre-commit --hook-type pre-push

# Sync the project with the venv
sync:
	@uv sync

# Sync the project with dev dependencies
sync-dev:
	@uv sync --all-extras

# Build wheel
build:
	@uv build

# Test
test:
	@uv run pytest

# Allure report
allure:
	@allure serve allure-results

# Ruff
ruff: fmt lint

# Lint only
lint:
	@uvx ruff check . --fix

# Format only
fmt:
	@uvx ruff format .

# Type check
type:
	@uvx mypy .

# Build image
image:
	@docker image build -t $(APP_NAME) .

# Start a compose service
compose-up:
	@docker compose -f ./compose.yml -p $(APP_NAME) up -d

# Shutdown a compose service
compose-down:
	@docker compose -f ./compose.yml down

# Clean build artifacts
clean:
	@rm -rf build dist *.egg-info htmlcov .coverage coverage.xml coverage.lcov
	@docker compose -f ./compose.yml down -v
	@docker image rm -f $(APP_NAME)

# Show help
help:
	@echo ""
	@echo "Usage:"
	@echo "    make [target]"
	@echo ""
	@echo "Targets:"
	@awk '/^[a-zA-Z\-_0-9]+:/ \
	{ \
		helpMessage = match(lastLine, /^# (.*)/); \
		if (helpMessage) { \
			helpCommand = substr($$1, 0, index($$1, ":")-1); \
			helpMessage = substr(lastLine, RSTART + 2, RLENGTH); \
			printf "\033[36m%-22s\033[0m %s\n", helpCommand,helpMessage; \
		} \
	} { lastLine = $$0 }' $(MAKEFILE_LIST)
