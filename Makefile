.PHONY: help install dev test lint typecheck format docker run clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -e .

dev: ## Install development dependencies and pre-commit hooks
	pip install -e ".[dev]"
	pre-commit install

test: ## Run all tests
	pytest tests/ -v --tb=short

lint: ## Run ruff linter
	ruff check .

typecheck: ## Run mypy type checker
	mypy core/ --ignore-missing-imports

format: ## Format code with ruff
	ruff format .
	ruff check --fix .

docker: ## Build Docker image
	docker build -t tokenmesh:latest .

run: ## Run the application locally
	uvicorn core.main:app --host 0.0.0.0 --port 8787 --reload

clean: ## Remove build artifacts and caches
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
