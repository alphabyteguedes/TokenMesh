from __future__ import annotations

"""End-to-End integration tests for TokenMesh dual-protocol proxy routes.

Validates:
* GET /healthz health check
* POST /v1/chat/completions (OpenAI native cache miss and cache hit)
* POST /v1/messages (Anthropic native cache miss and cache hit)
* Cross-provider translation (calling /v1/chat/completions with claude-3-5-sonnet-latest)
* Zero-trust DLP redaction headers
* GET /v1/analytics/summary (default USD and BRL currency conversion)
"""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

# Mock Responses
MOCK_OPENAI_RESPONSE = {
    "id": "chatcmpl-test-12345",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from mocked OpenAI!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
}

MOCK_ANTHROPIC_RESPONSE = {
    "id": "msg_test_67890",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello from mocked Anthropic!"}],
    "model": "claude-3-5-sonnet-latest",
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 12, "output_tokens": 6},
}


@pytest.mark.asyncio
async def test_healthz(client: AsyncClient) -> None:
    """GET /healthz returns 200 OK and healthy status."""
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"


@pytest.mark.asyncio
async def test_openai_chat_completion_cache_miss(client: AsyncClient) -> None:
    """POST /v1/chat/completions on first request yields a cache MISS and telemetry headers."""
    with patch(
        "core.api.routes_openai.openai_provider.chat_completion",
        new_callable=AsyncMock,
        return_value=MOCK_OPENAI_RESPONSE,
    ):
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Tell me a unique science fact #1"}],
        }
        resp = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer sk-test-key"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello from mocked OpenAI!"

        # Verify telemetry headers
        assert resp.headers["X-TokenMesh-Cache"] == "MISS"
        assert resp.headers["X-TokenMesh-Provider"] == "openai"
        assert "X-TokenMesh-Latency-Ms" in resp.headers


@pytest.mark.asyncio
async def test_openai_chat_completion_cache_hit(client: AsyncClient) -> None:
    """Repeating the identical request yields a cache HIT with sub-millisecond retrieval."""
    with patch(
        "core.api.routes_openai.openai_provider.chat_completion",
        new_callable=AsyncMock,
        return_value=MOCK_OPENAI_RESPONSE,
    ):
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Explain general relativity simply #2"}],
        }
        # Request 1: Miss
        resp1 = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer sk-test-key"},
        )
        assert resp1.status_code == 200
        assert resp1.headers["X-TokenMesh-Cache"] == "MISS"

        # Request 2: Hit
        resp2 = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer sk-test-key"},
        )
        assert resp2.status_code == 200
        assert resp2.headers["X-TokenMesh-Cache"] == "HIT"
        data2 = resp2.json()
        assert data2["choices"][0]["message"]["content"] == "Hello from mocked OpenAI!"


@pytest.mark.asyncio
async def test_anthropic_messages_cache_miss(client: AsyncClient) -> None:
    """POST /v1/messages yields a valid Anthropic response with cache MISS headers."""
    with patch(
        "core.api.routes_anthropic.anthropic_provider.chat_completion",
        new_callable=AsyncMock,
        return_value=MOCK_ANTHROPIC_RESPONSE,
    ):
        payload = {
            "model": "claude-3-5-sonnet-latest",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello Claude #3"}],
        }
        resp = await client.post(
            "/v1/messages",
            json=payload,
            headers={
                "x-api-key": "sk-ant-test-key",
                "anthropic-version": "2023-06-01",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert data["content"][0]["text"] == "Hello from mocked Anthropic!"
        assert resp.headers["X-TokenMesh-Cache"] == "MISS"
        assert resp.headers["X-TokenMesh-Provider"] == "anthropic"


@pytest.mark.asyncio
async def test_anthropic_messages_cache_hit(client: AsyncClient) -> None:
    """Repeating the Anthropic messages request yields a cache HIT."""
    with patch(
        "core.api.routes_anthropic.anthropic_provider.chat_completion",
        new_callable=AsyncMock,
        return_value=MOCK_ANTHROPIC_RESPONSE,
    ):
        payload = {
            "model": "claude-3-5-sonnet-latest",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Cache hit test for Claude #4"}],
        }
        resp1 = await client.post(
            "/v1/messages",
            json=payload,
            headers={
                "x-api-key": "sk-ant-test-key",
                "anthropic-version": "2023-06-01",
            },
        )
        assert resp1.status_code == 200
        assert resp1.headers["X-TokenMesh-Cache"] == "MISS"

        resp2 = await client.post(
            "/v1/messages",
            json=payload,
            headers={
                "x-api-key": "sk-ant-test-key",
                "anthropic-version": "2023-06-01",
            },
        )
        assert resp2.status_code == 200
        assert resp2.headers["X-TokenMesh-Cache"] == "HIT"
        data2 = resp2.json()
        assert data2["content"][0]["text"] == "Hello from mocked Anthropic!"


@pytest.mark.asyncio
async def test_cross_provider_claude_via_openai_route(client: AsyncClient) -> None:
    """Calling /v1/chat/completions with claude-* model translates to Anthropic and returns OpenAI schema."""
    with patch(
        "core.api.routes_openai.anthropic_provider.chat_completion",
        new_callable=AsyncMock,
        return_value=MOCK_ANTHROPIC_RESPONSE,
    ):
        payload = {
            "model": "claude-3-5-sonnet-latest",
            "messages": [{"role": "user", "content": "Cross provider test #5"}],
        }
        resp = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer sk-ant-test-key"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Hello from mocked Anthropic!"
        assert resp.headers["X-TokenMesh-Provider"] == "anthropic"


@pytest.mark.asyncio
async def test_dlp_redaction_in_request(client: AsyncClient) -> None:
    """Prompt containing sensitive data (CPF and Email) increments DLP redaction count."""
    with patch(
        "core.api.routes_openai.openai_provider.chat_completion",
        new_callable=AsyncMock,
        return_value=MOCK_OPENAI_RESPONSE,
    ):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": "My CPF is 123.456.789-00 and email is secret@example.com #6",
                }
            ],
        }
        resp = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer sk-test-key"},
        )
        assert resp.status_code == 200
        redactions = int(resp.headers.get("X-TokenMesh-DLP-Redactions", 0))
        assert redactions >= 2


@pytest.mark.asyncio
async def test_finops_summary_default_usd(client: AsyncClient) -> None:
    """GET /v1/analytics/summary returns zeroed/aggregated summary in USD."""
    resp = await client.get("/v1/analytics/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["currency"] == "USD"
    assert data["currency_symbol"] == "$"
    assert "total_requests" in data
    assert "kpi" in data


@pytest.mark.asyncio
async def test_finops_summary_brl(client: AsyncClient) -> None:
    """GET /v1/analytics/summary?currency=BRL converts currency and symbol to R$."""
    resp = await client.get("/v1/analytics/summary?currency=BRL")
    assert resp.status_code == 200
    data = resp.json()
    assert data["currency"] == "BRL"
    assert data["currency_symbol"] == "R$"
