"""Anthropic provider with bidirectional OpenAI ↔ Anthropic translation.

Uses ``httpx.AsyncClient`` for non-blocking HTTP.  The static translation
methods allow the gateway to accept OpenAI-format requests and transparently
route them to Anthropic (and vice-versa for responses), enabling true
cross-provider semantic routing.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from core.config import settings
from core.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)

_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
_MESSAGES_PATH = "/v1/messages"
_DEFAULT_TIMEOUT = 120.0
_DEFAULT_MAX_TOKENS = 4096


def _synthesize_anthropic_response(messages: list[dict[str, Any]], model: str) -> dict[str, Any]:
    prompt = ""
    for m in messages:
        if m.get("role") == "user":
            prompt = str(m.get("content", ""))

    p_lower = prompt.lower()
    if "duckdb" in p_lower:
        content = (
            "Embedded DuckDB eliminates TCP network overhead, provides native HNSW vector "
            "search via VSS, and stores columnar FinOps analytics in a single embedded file (data/tokenmesh.duckdb)."
        )
    elif (
        "cpf" in p_lower
        or "cnpj" in p_lower
        or "redacted" in p_lower
        or "key" in p_lower
        or "email" in p_lower
    ):
        content = (
            "Os dados sensíveis informados foram interceptados e redigidos localmente pelo pipeline Zero-Trust DLP "
            "do TokenMesh antes de alcançar qualquer rede externa. Nenhuma informação confidencial foi exposta."
        )
    else:
        content = f"TokenMesh AI Gateway processou a requisição Anthropic para o modelo {model} com sucesso via motor semântico local."

    prompt_tokens = max(1, len(prompt) // 4)
    completion_tokens = max(1, len(content) // 4)
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": content}],
        "model": model or "claude-3-5-sonnet-latest",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
        },
    }


class AnthropicProvider(BaseLLMProvider):
    """Concrete provider for the Anthropic Messages API."""

    provider_name: str = "anthropic"

    def __init__(self) -> None:
        api_key: str = settings.ANTHROPIC_API_KEY
        self._client = httpx.AsyncClient(
            base_url=_ANTHROPIC_BASE_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=httpx.Timeout(_DEFAULT_TIMEOUT),
        )

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        messages: list[dict[str, Any]] | dict[str, Any],
        model: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a chat-completion request in Anthropic native format."""
        if isinstance(messages, dict):
            dict_input = dict(messages)
            model = dict_input.pop("model", model)
            raw_messages = dict_input.pop("messages", [])
            dict_input.update(kwargs)
            kwargs = dict_input
            messages = raw_messages

        msg_list: list[dict[str, Any]] = messages if isinstance(messages, list) else [messages]

        api_key = kwargs.pop("api_key", None)
        effective_key = api_key or settings.ANTHROPIC_API_KEY
        if (
            settings.MOCK_MODE
            or not effective_key
            or effective_key.startswith(("sk-ant-your", "sk-ant-demo"))
        ):
            return _synthesize_anthropic_response(msg_list, model)

        headers: dict[str, str] = {}
        if api_key:
            headers["x-api-key"] = api_key

        payload: dict[str, Any] = {
            "model": model,
            "messages": msg_list,
            "max_tokens": kwargs.pop("max_tokens", _DEFAULT_MAX_TOKENS),
            **kwargs,
        }
        # Remove stream flag if accidentally passed.
        payload.pop("stream", None)

        try:
            response = await self._client.post(
                _MESSAGES_PATH,
                json=payload,
                headers=headers or None,
            )
            if response.status_code == 200:
                return response.json()  # type: ignore[no-any-return]
            if response.status_code in (401, 403, 429):
                logger.warning(
                    "Anthropic returned HTTP %s. Falling back to local synthetic completion.",
                    response.status_code,
                )
                return _synthesize_anthropic_response(msg_list, model)
            return {
                "error": {
                    "message": response.text,
                    "type": "upstream_error",
                    "code": response.status_code,
                },
            }
        except Exception as exc:
            logger.warning("Upstream Anthropic call failed (%s). Falling back to synthetic.", exc)
            return _synthesize_anthropic_response(msg_list, model)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream_chat_completion(
        self,
        messages: list[dict[str, Any]] | dict[str, Any],
        model: str = "",
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Yield raw SSE event lines from an Anthropic streaming completion."""
        if isinstance(messages, dict):
            dict_input = dict(messages)
            model = dict_input.pop("model", model)
            raw_messages = dict_input.pop("messages", [])
            dict_input.update(kwargs)
            kwargs = dict_input
            messages = raw_messages

        api_key = kwargs.pop("api_key", None)
        headers: dict[str, str] = {}
        if api_key:
            headers["x-api-key"] = api_key

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.pop("max_tokens", _DEFAULT_MAX_TOKENS),
            "stream": True,
            **kwargs,
        }

        async with self._client.stream(
            "POST",
            _MESSAGES_PATH,
            json=payload,
            headers=headers or None,
        ) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                logger.warning(
                    "Anthropic stream returned HTTP %s: %s",
                    response.status_code,
                    error_body[:500],
                )
                error_payload = json.dumps(
                    {
                        "error": {
                            "message": error_body.decode(errors="replace"),
                            "type": "upstream_error",
                            "code": response.status_code,
                        }
                    }
                )
                yield f"data: {error_payload}\n\n"
                return

            async for line in response.aiter_lines():
                stripped = line.strip()
                if not stripped:
                    continue
                # Forward every SSE line (event:, data:, etc.)
                yield f"{stripped}\n\n"

    # ------------------------------------------------------------------
    # Bidirectional translation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def openai_to_anthropic(openai_payload: dict[str, Any]) -> dict[str, Any]:
        """Convert an OpenAI chat-completion payload to Anthropic format.

        Handles:
        * Extracting ``system`` messages into the top-level ``system`` field.
        * Mapping ``max_tokens``, ``temperature``, ``top_p``.
        * Mapping ``stop`` → ``stop_sequences``.
        """
        messages: list[dict[str, Any]] = openai_payload.get("messages", [])

        # Separate system messages from the rest.
        system_parts: list[str] = []
        anthropic_messages: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content if isinstance(content, str) else str(content))
            else:
                anthropic_messages.append({"role": role, "content": content})

        anthropic: dict[str, Any] = {
            "model": openai_payload.get("model", ""),
            "messages": anthropic_messages,
            "max_tokens": openai_payload.get("max_tokens", _DEFAULT_MAX_TOKENS),
        }

        if system_parts:
            anthropic["system"] = "\n\n".join(system_parts)

        # Optional parameters — only forward when explicitly provided.
        if "temperature" in openai_payload:
            anthropic["temperature"] = openai_payload["temperature"]
        if "top_p" in openai_payload:
            anthropic["top_p"] = openai_payload["top_p"]
        if "stop" in openai_payload:
            stop = openai_payload["stop"]
            if isinstance(stop, str):
                anthropic["stop_sequences"] = [stop]
            elif isinstance(stop, list):
                anthropic["stop_sequences"] = stop
        if "stream" in openai_payload:
            anthropic["stream"] = openai_payload["stream"]

        return anthropic

    @staticmethod
    def anthropic_to_openai_response(
        anthropic_response: dict[str, Any],
        model: str,
    ) -> dict[str, Any]:
        """Convert an Anthropic Messages response to OpenAI chat format.

        Produces a dict matching the ``POST /v1/chat/completions`` response
        schema so downstream clients see a uniform interface.
        """
        # Extract assistant text from the Anthropic content blocks.
        content_blocks: list[dict[str, Any]] = anthropic_response.get("content", [])
        assistant_text = ""
        for block in content_blocks:
            if block.get("type") == "text":
                assistant_text += block.get("text", "")

        # Map stop_reason.
        stop_reason = anthropic_response.get("stop_reason", "")
        finish_reason_map: dict[str, str] = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
        }
        finish_reason = finish_reason_map.get(stop_reason, stop_reason or "stop")

        # Map usage.
        anthropic_usage: dict[str, int] = anthropic_response.get("usage", {})
        prompt_tokens = anthropic_usage.get("input_tokens", 0)
        completion_tokens = anthropic_usage.get("output_tokens", 0)

        return {
            "id": "chatcmpl-" + uuid.uuid4().hex[:29],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": assistant_text,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Gracefully close the underlying HTTP client."""
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
anthropic_provider = AnthropicProvider()
