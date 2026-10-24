# Routing Cursor IDE through TokenMesh

Configure [Cursor IDE](https://www.cursor.com) to route all inline edits, chat, and agentic workflows through **TokenMesh**. This saves up to 70% of OpenAI & Anthropic API costs through local semantic caching and automatically sanitizes confidential code patterns (API keys, credentials, PII) using local DLP.

---

## 1. Start the TokenMesh Gateway

Make sure TokenMesh is running locally:

```bash
tokenmesh start --port 8787
```

Confirm health at [http://localhost:8787/healthz](http://localhost:8787/healthz).

---

## 2. Cursor Settings Configuration

1. Open Cursor and navigate to **Cursor Settings** (`Ctrl + Shift + J` or `Cmd + Shift + J`).
2. Go to the **Models** tab in the sidebar.
3. Scroll down to **OpenAI API Key**:
   - Toggle **Override OpenAI Base URL**: `ON`
   - Set **Base URL** to:
     ```text
     http://localhost:8787/v1
     ```
   - Enter your real `sk-...` API key (or leave `tm-local-key` if you have `OPENAI_API_KEY` defined in TokenMesh `.env`).
4. (Optional) Under **Anthropic API Key**:
   - TokenMesh provides native cross-provider routing. When Cursor requests `claude-3-5-sonnet`, TokenMesh translates the request and proxies it upstream.

---

## 3. Verify in TokenMesh Dashboard

1. Ask Cursor Composer or Cursor Chat a question.
2. Ask a similar or repeated question (e.g. asking to explain the same function or generate unit tests).
3. Open [http://localhost:8787/dashboard](http://localhost:8787/dashboard) to view:
   - **Semantic Cache Hits** with sub-5ms response times.
   - **Financial Savings** converted to your preferred currency (USD, BRL, EUR, GBP, JPY).
   - **DLP Redactions** applied to your codebase before requests hit upstream APIs.
