"""Base interface for LLM providers.

Every concrete provider (OpenAI, Anthropic, …) inherits from
:class:`BaseLLMProvider` and implements the two abstract methods for
regular and streaming chat completions.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

from core.config import settings

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class that all LLM providers must implement."""

    provider_name: str = "base"

    # -- abstract methods ---------------------------------------------------

    @abstractmethod
    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a chat-completion request and return a normalised response dict."""
        ...

    @abstractmethod
    async def stream_chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Yield Server-Sent Event (SSE) chunks from a streaming completion."""
        ...
        # Make the function a generator even though the body is abstract.
        yield ""  # pragma: no cover

    # -- shared helpers -----------------------------------------------------

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate the USD cost for a completion using configured model prices."""
        return settings.get_model_cost(model, input_tokens, output_tokens)
