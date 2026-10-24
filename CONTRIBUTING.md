# Contributing to TokenMesh

Thank you for your interest in contributing to TokenMesh! This guide will help you get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Architecture Overview](#architecture-overview)
- [How to Contribute](#how-to-contribute)
- [Adding a New Provider](#adding-a-new-provider)
- [Adding a New DLP Rule](#adding-a-new-dlp-rule)
- [Commit Conventions](#commit-conventions)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code.

## Getting Started

### Prerequisites

- Python 3.11+
- Git
- Make (optional, but recommended)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/tokenmesh/tokenmesh.git
cd tokenmesh

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install development dependencies
make dev
# or: pip install -e ".[dev]"

# Copy environment file
cp .env.example .env
# Edit .env with your API keys

# Run the application
make run

# Run tests
make test
```

## Architecture Overview

TokenMesh follows a modular architecture designed for parallel development:

```
core/
├── api/          # HTTP route handlers (one file per protocol)
├── db/           # DuckDB engine (semantic cache + analytics)
├── pipeline/     # Request processing pipeline (DLP, compression, embedding)
├── providers/    # LLM provider clients (one file per provider)
└── schemas/      # Pydantic v2 data models
```

### Key Design Principles

1. **Zero External Dependencies for Core**: DuckDB handles both vector search and analytics. No Redis, no Postgres, no cloud vector DBs.
2. **Protocol-Native**: Each LLM protocol (OpenAI, Anthropic) has its own route handler and schema, with a canonical normalization layer for cross-protocol caching.
3. **Pipeline Modularity**: DLP, compression, and embedding are independent modules that can be enabled/disabled via configuration.
4. **Thread-Safe Async**: All blocking operations (DuckDB, embedding) run in `asyncio.to_thread()` with write locks where needed.

## How to Contribute

### Reporting Bugs

Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.yml) to file bugs. Include:
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, TokenMesh version)

### Suggesting Features

Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.yml).

### Adding a New Provider

To add support for a new LLM provider (e.g., Google Gemini, Mistral):

1. Create `core/providers/your_provider.py` implementing `BaseLLMProvider`
2. Add the provider's models and pricing to `core/config.py` → `PRICING_TABLE`
3. Create a new route file `core/api/routes_yourprovider.py` if the provider has a unique API contract
4. Register the route in `core/main.py`
5. Add tests in `tests/test_yourprovider.py`
6. Update `CODEOWNERS` to assign reviewers

```python
# Example: core/providers/gemini_provider.py
from core.providers.base import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    provider_name = "gemini"

    async def chat_completion(self, messages, model, **kwargs):
        # Implement Gemini API call
        ...

    async def stream_chat_completion(self, messages, model, **kwargs):
        # Implement Gemini streaming
        ...
```

### Adding a New DLP Rule

To add a new data sanitization pattern:

1. Edit `core/pipeline/dlp.py`
2. Add a new `DLPPattern` to the `DLP_PATTERNS` list (maintain specificity order)
3. Add tests in `tests/test_dlp_and_compressor.py`

```python
# Example: Adding phone number redaction
(
    DLPPattern(
        name="PHONE_BR",
        pattern=re.compile(r"\b\(?\d{2}\)?\s?\d{4,5}-?\d{4}\b"),
        replacement="<REDACTED_PHONE>",
    ),
)
```

## Commit Conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(providers): add Google Gemini provider
fix(cache): handle empty embedding vectors
docs(readme): add Claude Code configuration example
test(dlp): add phone number redaction tests
perf(db): optimize HNSW index parameters
ci: update Python version in CI pipeline
```

### Types: `feat`, `fix`, `docs`, `test`, `perf`, `ci`, `refactor`, `chore`
### Scopes: `providers`, `cache`, `dlp`, `db`, `api`, `pipeline`, `web`, `config`

## Pull Request Process

1. Fork the repository and create your branch from `main`
2. Follow the [PR template](.github/PULL_REQUEST_TEMPLATE.md)
3. Ensure all checks pass: `make lint && make typecheck && make test`
4. Update documentation if you're changing behavior
5. Request review from the appropriate code owners (see `CODEOWNERS`)

### Latency Benchmark

For performance-sensitive changes, include a latency benchmark:

```bash
# Run the benchmark
python -m pytest tests/ -v --tb=short -k "test_" --durations=10
```

Report the P50/P99 latency for cache hit and miss scenarios in your PR description.
