# TokenMesh Air-Gapped & Offline Deployment Guide

TokenMesh was architected from day one with a **Zero-Cloud-Dependency** design principle. It does not require any external vector database SaaS (Pinecone, Qdrant, Weaviate) or cloud cache (Redis, Upstash).

This guide explains how to deploy TokenMesh inside **100% isolated, air-gapped environments** (e.g. banking intranets, government networks, or defense enclaves) where outbound internet access is strictly prohibited.

---

## 1. Prerequisites to Pre-Package Offline

In an environment without internet access, two components must be pre-bundled into your container or deployment artifact:
1. **FastEmbed ONNX Model Weights** (`BAAI/bge-small-en-v1.5`, ~133 MB).
2. **DuckDB VSS Extension Binary** (`vss.duckdb_extension`).

---

## 2. Pre-Caching FastEmbed Weights

On an internet-connected build machine, run a one-line script to download the ONNX model files:

```python
from fastembed import TextEmbedding

# Downloads model to ~/.cache/fastembed/
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
```

The model weights will be saved to:
- Linux / macOS: `~/.cache/fastembed/`
- Windows: `%USERPROFILE%\.cache\fastembed\`

### Setting Cache Path in Production
Copy the cached directory to your air-gapped server (e.g. `/opt/tokenmesh/models/`) and set the environment variable:

```bash
export FASTEMBED_CACHE_PATH="/opt/tokenmesh/models"
```

TokenMesh will read embeddings directly from the local filesystem with zero network attempts.

---

## 3. Pre-Installing DuckDB VSS Extension

DuckDB extensions can be pre-downloaded and stored in the DuckDB extension directory.

### Step A: Download the extension binary on a connected machine
```bash
duckdb -c "INSTALL vss;"
```
DuckDB saves downloaded extensions to:
- Linux: `~/.duckdb/extensions/<duckdb_version>/<platform>/vss.duckdb_extension`
- Windows: `%APPDATA%\duckdb\extensions\<duckdb_version>\<platform>\vss.duckdb_extension`

### Step B: Transfer to Air-Gapped Target
Copy the `~/.duckdb/extensions/` folder to the target server's user home directory or inside the Docker image:

```dockerfile
# Inside air-gapped Dockerfile
COPY ./offline_extensions/ /root/.duckdb/extensions/
COPY ./offline_models/     /root/.cache/fastembed/
```

When TokenMesh boots, DuckDB will execute `LOAD vss;` directly from the local filesystem without making any HTTP requests.

---

## 4. Local Upstream Models (vLLM / Ollama / LocalAI)

In an air-gapped environment, LLM inference is typically hosted on local GPU clusters using [vLLM](https://github.com/vllm-project/vllm), [Ollama](https://ollama.ai), or [TGI](https://github.com/huggingface/text-generation-inference).

TokenMesh natively proxies any OpenAI-compatible or Anthropic-compatible server by configuring:

```env
# Point upstream to internal private vLLM or Ollama instance
OPENAI_BASE_URL="http://vllm-cluster.internal:8000/v1"
```

All local client requests will flow through TokenMesh, providing:
1. **Local DLP Sanitization:** PII and credentials redacted before reaching internal models.
2. **Sub-5ms Semantic Caching:** Redundant prompts answered immediately by DuckDB without consuming GPU compute.
3. **FinOps & Usage Telemetry:** Complete audit ledger of all organizational queries.
