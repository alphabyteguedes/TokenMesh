# Routing Claude Code CLI through TokenMesh

Configure the official **Claude Code** CLI (`@anthropic-ai/claude-code`) to route all terminal prompts, indexing passes, and tool execution queries through **TokenMesh**.

---

## 1. Start the TokenMesh Gateway

```bash
tokenmesh start --port 8787
```

Confirm health at [http://localhost:8787/healthz](http://localhost:8787/healthz).

---

## 2. Set Environment Variables

Claude Code natively respects standard Anthropic environment variables. Simply point `ANTHROPIC_BASE_URL` to TokenMesh:

### Linux / macOS:
```bash
export ANTHROPIC_BASE_URL="http://localhost:8787"
export ANTHROPIC_API_KEY="sk-ant-your-key-here"  # TokenMesh will forward or inject from .env
```

### Windows (PowerShell):
```powershell
$env:ANTHROPIC_BASE_URL = "http://localhost:8787"
$env:ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```

### Windows (CMD):
```cmd
set ANTHROPIC_BASE_URL=http://localhost:8787
set ANTHROPIC_API_KEY=sk-ant-your-key-here
```

---

## 3. Run Claude Code

Execute Claude Code normally in your repository:

```bash
claude
```

TokenMesh intercepts every prompt:
- **Identical & Semantically Similar Prompts:** Served instantly from DuckDB VSS HNSW in sub-5ms, returning standard Anthropic SSE streaming events (`event: content_block_delta`, etc.).
- **FinOps Ledger:** Open [http://localhost:8787/dashboard](http://localhost:8787/dashboard) to track tokens saved and multi-currency cost reductions in real time.
- **Sensitive Data Protection:** API keys and credentials detected in terminal context are automatically sanitized before leaving your machine.
