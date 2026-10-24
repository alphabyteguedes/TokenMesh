"""OpenAI provider — forwards requests to the OpenAI chat-completions API.

Uses ``httpx.AsyncClient`` for non-blocking HTTP with connection pooling.
Streaming responses are forwarded as raw SSE lines so the gateway can
relay them to downstream clients with minimal latency.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from core.config import settings
from core.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)

_OPENAI_BASE_URL = "https://api.openai.com"
_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
_DEFAULT_TIMEOUT = 120.0


class OpenAIProvider(BaseLLMProvider):
    """Concrete provider for the OpenAI API."""

    provider_name: str = "openai"

    def __init__(self) -> None:
        api_key: str = settings.OPENAI_API_KEY
        self._client = httpx.AsyncClient(
            base_url=_OPENAI_BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(_DEFAULT_TIMEOUT),
        )

    # -- non-streaming ------------------------------------------------------

    async def chat_completion(
        self,
        messages: list[dict[str, Any]] | dict[str, Any],
        model: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a blocking chat-completion request and return the JSON body."""
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
            headers["Authorization"] = f"Bearer {api_key}"

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            **kwargs,
        }
        # Ensure we don't accidentally request a stream here.
        payload.pop("stream", None)

        response = await self._client.post(
            _CHAT_COMPLETIONS_PATH,
            json=payload,
            headers=headers or None,
        )

        if response.status_code != 200:
            logger.warning(
                "OpenAI returned HTTP %s: %s",
                response.status_code,
                response.text[:500],
            )
            return {
                "error": {
                    "message": response.text,
                    "type": "upstream_error",
                    "code": response.status_code,
                },
            }

        return response.json()  # type: ignore[no-any-return]

    # -- streaming ----------------------------------------------------------

    async def stream_chat_completion(
        self,
        messages: list[dict[str, Any]] | dict[str, Any],
        model: str = "",
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Yield raw SSE ``data: {…}`` lines from a streaming completion."""
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
            headers["Authorization"] = f"Bearer {api_key}"

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            **kwargs,
        }

        async with self._client.stream(
            "POST",
            _CHAT_COMPLETIONS_PATH,
            json=payload,
            headers=headers or None,
        ) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                logger.warning(
                    "OpenAI stream returned HTTP %s: %s",
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
                # Forward SSE data lines verbatim.
                if stripped.startswith("data:"):
                    yield f"{stripped}\n\n"

    # -- lifecycle ----------------------------------------------------------

    async def close(self) -> None:
        """Gracefully close the underlying HTTP client."""
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
openai_provider = OpenAIProvider()
