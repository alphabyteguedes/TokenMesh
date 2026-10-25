<p align="center">
  <h1 align="center">🔷 TokenMesh</h1>
  <p align="center"><strong>AI Gateway & Semantic Reverse Proxy</strong></p>
  <p align="center">
    <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
    <img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white">
    <a href="https://github.com/alphabyteguedes/TokenMesh/actions"><img alt="CI Status" src="https://img.shields.io/github/actions/workflow/status/alphabyteguedes/TokenMesh/ci.yml?branch=main&label=CI"></a>
    <a href="https://hub.docker.com/r/alphabyteguedes/tokenmesh"><img alt="Docker" src="https://img.shields.io/docker/v/alphabyteguedes/tokenmesh?label=Docker&logo=docker"></a>
  </p>
</p>

---

## What is TokenMesh?

**TokenMesh** is a self-hosted AI gateway and semantic reverse proxy that sits between your applications and LLM providers. It **reduces API costs by up to 70%** through semantic caching powered by DuckDB, applies **local Data Loss Prevention (DLP)** to protect sensitive data, and supports **OpenAI and Anthropic protocols natively** — meaning your existing SDKs, tools, and agents work without any code changes.

### Key Differentiators

| Feature | TokenMesh | Typical Gateways |
|---------|-----------|------------------|
| **Cloud Dependencies** | Zero — everything runs locally | Redis, Postgres, Pinecone, etc. |
| **Overhead Latency** | Sub-5ms on cache hit | 20–100ms+ |
| **Protocol Support** | Dual-native (OpenAI + Anthropic) | Usually OpenAI-only |
| **Embedding** | Local model (all-MiniLM-L6-v2) | Cloud embedding APIs |
| **DLP** | Built-in, zero-trust | External service or none |
| **Analytics** | FinOps dashboard included | Separate tool required |

---

## Features

- 🧠 **Semantic Cache** — DuckDB + HNSW vector similarity search. Semantically identical prompts return cached responses in under 5ms.
- 🔀 **Dual Protocol** — Native support for both OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) API contracts.
- 🔄 **Cross-Provider Routing** — Use `claude-*` models through the OpenAI SDK. TokenMesh automatically translates between protocols.
- 🛡️ **Zero-Trust DLP** — Automatically redacts CPF, CNPJ, email addresses, credit cards, and API keys from prompts before they reach any provider.
- 📊 **FinOps Analytics** — Real-time cost tracking with multi-currency support (USD, BRL, EUR). Know exactly what you're spending per model, per day.
- 🌍 **Multilingual Dashboard** — Built-in web dashboard in PT-BR, EN, and ES for monitoring cache performance, costs, and request analytics.
- 🐋 **Docker-Ready** — Single `docker compose up` to run the entire stack. No external services required.
- 📦 **Prompt Compression** — Optional prompt compression to reduce token usage by up to 40%.
- 🔑 **API Key Passthrough** — Your API keys stay local. TokenMesh never stores or transmits keys to third parties.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          CLIENTS                                 │
│  OpenAI SDK · Anthropic SDK · Cursor · Claude Code · LangChain  │
└──────────────────────┬───────────────────────────────────────────┘
                       │ HTTP (same API contract)
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                     TOKENMESH GATEWAY                            │
│                     localhost:8787                                │
│                                                                  │
│  ┌─────────┐   ┌────────────┐   ┌───────────┐   ┌───────────┐  │
│  │  Routes  │──▶│    DLP     │──▶│ Compress  │──▶│  Embed    │  │
│  │ OpenAI / │   │ Redact PII │   │ Reduce    │   │ Vectorize │  │
│  │ Anthropic│   │ & API Keys │   │ Tokens    │   │ Prompt    │  │
│  └─────────┘   └────────────┘   └───────────┘   └─────┬─────┘  │
│                                                        │         │
│                                                        ▼         │
│                                              ┌─────────────────┐ │
│                                              │   DuckDB Cache  │ │
│                                              │   HNSW Index    │ │
│                                              │   + Analytics   │ │
│                                              └────────┬────────┘ │
│                                                       │          │
│                                          Cache Hit?   │          │
│                                          ┌────────────┤          │
│                                          │ YES        │ NO       │
│                                          ▼            ▼          │
│                                    ┌──────────┐ ┌────────────┐  │
│                                    │  Return   │ │  Forward   │  │
│                                    │  Cached   │ │  to LLM    │  │
│                                    │  Response │ │  Provider   │  │
│                                    └──────────┘ └────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                                                        │
                       ┌────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                        LLM PROVIDERS                             │
│            OpenAI API  ·  Anthropic API  ·  (more soon)         │
└──────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Docker (Recommended)

```bash
git clone https://github.com/alphabyteguedes/TokenMesh.git
cd TokenMesh
cp .env.example .env
# Edit .env with your API keys
docker compose up -d

# Dashboard available at http://localhost:8787
```

### Local Installation

```bash
git clone https://github.com/alphabyteguedes/TokenMesh.git
cd TokenMesh
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install -e .
cp .env.example .env
# Edit .env with your API keys

python -m core.main
# Dashboard available at http://localhost:8787
```

---

## SDK Configuration Examples

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(api_key="your-openai-key", base_url="http://localhost:8787/v1")

response = client.chat.completions.create(
    model="gpt-4o", messages=[{"role": "user", "content": "Hello!"}]
)
```

### OpenAI SDK → Claude (Cross-Provider Routing)

Use Claude models through the OpenAI SDK — TokenMesh translates the protocol automatically:

```python
from openai import OpenAI

client = OpenAI(api_key="your-anthropic-key", base_url="http://localhost:8787/v1")

response = client.chat.completions.create(
    model="claude-3-5-sonnet-latest", messages=[{"role": "user", "content": "Hello!"}]
)
```

### Anthropic Python SDK

```python
import anthropic

client = anthropic.Anthropic(api_key="your-anthropic-key", base_url="http://localhost:8787")

message = client.messages.create(
    model="claude-3-5-sonnet-latest",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### Claude Code

```bash
export ANTHROPIC_BASE_URL=http://localhost:8787
claude code
```

### Cursor

In Cursor settings, set the API base URL to:

```
http://localhost:8787/v1
```

### LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o", openai_api_key="your-key", openai_api_base="http://localhost:8787/v1"
)
```

---

## Telemetry Headers

Every response from TokenMesh includes telemetry headers so you can observe caching and cost behavior:

| Header | Description |
|--------|-------------|
| `X-TokenMesh-Cache` | `HIT` or `MISS` — whether the response was served from cache |
| `X-TokenMesh-Similarity` | Cosine similarity score (0.0–1.0) of the cache match |
| `X-TokenMesh-Tokens-Saved` | Number of tokens saved by serving from cache |
| `X-TokenMesh-Cost-USD` | Estimated cost of the request in USD |
| `X-TokenMesh-Cost-Local` | Estimated cost in local currency (configured via `TOKENMESH_LOCAL_CURRENCY`) |
| `X-TokenMesh-DLP-Redacted` | Number of PII fields redacted by the DLP pipeline |
| `X-TokenMesh-Processing-Ms` | Total gateway processing time in milliseconds |
| `X-TokenMesh-Provider` | The upstream provider that handled the request (`openai`, `anthropic`) |
| `X-TokenMesh-Model` | The actual model used for the request |

---

## Configuration

TokenMesh is configured via environment variables. Create a `.env` file from `.env.example`:

| Variable | Default | Description |
|----------|---------|-------------|
| `TOKENMESH_PORT` | `8787` | HTTP server port |
| `TOKENMESH_HOST` | `0.0.0.0` | HTTP server bind address |
| `TOKENMESH_LOG_LEVEL` | `info` | Logging level (`debug`, `info`, `warning`, `error`) |
| `TOKENMESH_DB_PATH` | `./data/tokenmesh.duckdb` | Path to the DuckDB database file |
| `TOKENMESH_CACHE_ENABLED` | `true` | Enable/disable semantic cache |
| `TOKENMESH_CACHE_THRESHOLD` | `0.92` | Cosine similarity threshold for cache hits (0.0–1.0) |
| `TOKENMESH_CACHE_TTL_HOURS` | `168` | Cache entry time-to-live in hours (default: 7 days) |
| `TOKENMESH_DLP_ENABLED` | `true` | Enable/disable DLP pipeline |
| `TOKENMESH_COMPRESS_ENABLED` | `false` | Enable/disable prompt compression |
| `TOKENMESH_EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model for embeddings |
| `TOKENMESH_LOCAL_CURRENCY` | `USD` | Local currency for cost display (`USD`, `BRL`, `EUR`) |
| `TOKENMESH_EXCHANGE_RATE` | `1.0` | Exchange rate from USD to local currency |
| `TOKENMESH_CORS_ORIGINS` | `*` | Allowed CORS origins (comma-separated) |
| `TOKENMESH_DASHBOARD_LANG` | `en` | Dashboard language (`en`, `pt-br`, `es`) |
| `OPENAI_API_KEY` | — | Your OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Your Anthropic API key |

---

## Development

```bash
make dev        # Install dev dependencies
make run        # Run locally with hot reload
make test       # Run tests
make lint       # Lint code (ruff)
make format     # Format code (ruff format)
make typecheck  # Type check (mypy)
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development guide.

---

## Contributing

We welcome contributions! Please read our [Contributing Guide](CONTRIBUTING.md) to learn about our development process, coding standards, and how to submit pull requests.

Before contributing, please review our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Security

To report a security vulnerability, please email **alphabyteltda@gmail.com**. See our [Security Policy](SECURITY.md) for details.

---

## License

TokenMesh is licensed under the [Apache License 2.0](LICENSE).

```
Copyright 2024 TokenMesh Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```
