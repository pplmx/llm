.PHONY: help build test test-fast test-quick test-cov test-e2e ruff image clean lint fmt dev
.DEFAULT_GOAL := help

APP_NAME := llm

# Init the venv
init: sync
	@uvx prek install --hook-type commit-msg --hook-type pre-commit --hook-type pre-push

# Sync the project with the venv
sync:
	@uv sync

# Sync the project with dev dependencies
dev:
	@uv sync --all-extras --all-groups

# Build wheel
build:
	@uv build

# Test all
test:
	@uv run pytest

# Test fast (exclude heavy and e2e, for daily development)
test-fast:
	@uv run pytest -m "not heavy and not e2e"

# Test quick only
test-quick:
	@uv run pytest -m quick

# Test with coverage and allure reports (for CI)
test-cov:
	@uv run pytest --cov=llm --cov-report=term-missing --cov-report=html --cov-report=lcov --cov-report=xml --alluredir=allure-results

# Test e2e only
test-e2e:
	@uv run pytest -m e2e

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

# ty type check
ty:
	@uvx ty check

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
