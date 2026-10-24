from __future__ import annotations

"""Pydantic v2 schemas for the OpenAI ``/v1/chat/completions`` protocol."""

from typing import Any

from pydantic import BaseModel


class OpenAIMessage(BaseModel):
    """A single message in the OpenAI chat format."""

    role: str
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class OpenAIChatRequest(BaseModel):
    """Request body for ``POST /v1/chat/completions``."""

    model: str
    messages: list[OpenAIMessage]
    temperature: float | None = None
    top_p: float | None = None
    n: int = 1
    stream: bool = False
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    user: str | None = None


class OpenAIUsage(BaseModel):
    """Token-usage counters returned by the OpenAI API."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIChoice(BaseModel):
    """A single completion choice."""

    index: int
    message: OpenAIMessage
    finish_reason: str | None = None


class OpenAIChatResponse(BaseModel):
    """Response body for ``POST /v1/chat/completions``."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChoice]
    usage: OpenAIUsage
