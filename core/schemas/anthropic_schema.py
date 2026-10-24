from __future__ import annotations

"""Pydantic v2 schemas for the Anthropic ``/v1/messages`` protocol."""

from typing import Any

from pydantic import BaseModel


class AnthropicContentBlock(BaseModel):
    """A single content block in an Anthropic message."""

    type: str = "text"
    text: str


class AnthropicMessage(BaseModel):
    """A single message in the Anthropic chat format."""

    role: str
    content: str | list[AnthropicContentBlock]


class AnthropicMessagesRequest(BaseModel):
    """Request body for ``POST /v1/messages``."""

    model: str
    messages: list[AnthropicMessage]
    max_tokens: int = 4096
    system: str | list[dict[str, Any]] | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    stop_sequences: list[str] | None = None
    metadata: dict[str, Any] | None = None


class AnthropicUsage(BaseModel):
    """Token-usage counters returned by the Anthropic API."""

    input_tokens: int
    output_tokens: int


class AnthropicMessagesResponse(BaseModel):
    """Response body for ``POST /v1/messages``."""

    id: str
    type: str = "message"
    role: str = "assistant"
    content: list[AnthropicContentBlock]
    model: str
    stop_reason: str | None = "end_turn"
    stop_sequence: str | None = None
    usage: AnthropicUsage
