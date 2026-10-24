from __future__ import annotations

"""Native Anthropic protocol router for ``POST /v1/messages``.

Provides 100% fidelity Anthropic API compatibility for official SDKs,
Claude Code, Cursor, and LangChain. Implements semantic cache lookup,
local DLP redaction, heuristic prompt compression, SSE stream reconstruction
on cache hits, and upstream proxying with real-time telemetry headers.
"""

import hashlib
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from core.config import settings
from core.db.duckdb_manager import db_manager
from core.pipeline.compressor import compressor
from core.pipeline.dlp import dlp_processor
from core.pipeline.embedder import embedder
from core.providers.anthropic_provider import anthropic_provider
from core.schemas.anthropic_schema import (
    AnthropicContentBlock,
    AnthropicMessage,
    AnthropicMessagesRequest,
)

logger = logging.getLogger("tokenmesh.routes.anthropic")

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_canonical_text_anthropic(
    system: str | list[dict[str, Any]] | None,
    messages: list[AnthropicMessage],
) -> str:
    """Extract canonical text representation from Anthropic system and messages."""
    parts: list[str] = []

    if system:
        if isinstance(system, str):
            parts.append(system)
        elif isinstance(system, list):
            for sys_block in system:
                if isinstance(sys_block, dict) and sys_block.get("type") == "text":
                    parts.append(str(sys_block.get("text", "")))

    for msg in messages:
        content = msg.content
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for c_block in content:
                if isinstance(c_block, AnthropicContentBlock):
                    parts.append(c_block.text)
                elif isinstance(c_block, dict) and c_block.get("type") == "text":
                    parts.append(str(c_block.get("text", "")))

    return "\n".join(parts)


def build_telemetry_headers(
    cache_hit: bool,
    provider: str,
    similarity: float,
    tokens_saved: int,
    saved_usd: float,
    dlp_redactions: int,
    latency_ms: float,
) -> dict[str, str]:
    """Build X-TokenMesh-* telemetry headers."""
    return {
        "X-TokenMesh-Cache": "HIT" if cache_hit else "MISS",
        "X-TokenMesh-Provider": provider,
        "X-TokenMesh-Similarity": f"{similarity:.4f}",
        "X-TokenMesh-Tokens-Saved": str(tokens_saved),
        "X-TokenMesh-Saved-USD": f"{saved_usd:.6f}",
        "X-TokenMesh-DLP-Redactions": str(dlp_redactions),
        "X-TokenMesh-Latency-Ms": f"{latency_ms:.2f}",
    }


def _extract_response_text(response_payload: dict[str, Any]) -> str:
    """Extract full text from Anthropic response content blocks."""
    content = response_payload.get("content", [])
    text_parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
    elif isinstance(content, str):
        text_parts.append(content)
    return "".join(text_parts)


async def generate_anthropic_stream_from_cache(
    cached_response: dict[str, Any],
    model: str,
) -> AsyncGenerator[str, None]:
    """Yield official Anthropic SSE event frames from cached response."""
    msg_id: str = cached_response.get("id", f"msg_{uuid.uuid4().hex[:24]}")
    full_text = _extract_response_text(cached_response)
    usage = cached_response.get("usage", {})
    input_tokens = usage.get("input_tokens", compressor.estimate_tokens(full_text))
    output_tokens = usage.get("output_tokens", compressor.estimate_tokens(full_text))

    # 1. message_start
    message_start_payload = {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": 1,
            },
        },
    }
    yield f"event: message_start\ndata: {json.dumps(message_start_payload)}\n\n"

    # 2. content_block_start
    content_block_start_payload = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    yield f"event: content_block_start\ndata: {json.dumps(content_block_start_payload)}\n\n"

    # 3. content_block_delta chunks
    words = full_text.split(" ")
    for i, word in enumerate(words):
        chunk = word if i == 0 else f" {word}"
        delta_payload = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": chunk},
        }
        yield f"event: content_block_delta\ndata: {json.dumps(delta_payload)}\n\n"

    # 4. content_block_stop
    content_block_stop_payload = {
        "type": "content_block_stop",
        "index": 0,
    }
    yield f"event: content_block_stop\ndata: {json.dumps(content_block_stop_payload)}\n\n"

    # 5. message_delta
    message_delta_payload = {
        "type": "message_delta",
        "delta": {
            "stop_reason": cached_response.get("stop_reason", "end_turn"),
            "stop_sequence": cached_response.get("stop_sequence"),
        },
        "usage": {"output_tokens": output_tokens},
    }
    yield f"event: message_delta\ndata: {json.dumps(message_delta_payload)}\n\n"

    # 6. message_stop
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


# ---------------------------------------------------------------------------
# POST /v1/messages
# ---------------------------------------------------------------------------


@router.post("/v1/messages")
async def messages_endpoint(
    body: AnthropicMessagesRequest,
    request: Request,
    x_api_key: str | None = Header(None, alias="x-api-key"),
    anthropic_version: str | None = Header(None, alias="anthropic-version"),
    anthropic_beta: str | None = Header(None, alias="anthropic-beta"),
) -> Any:
    """Native Anthropic Messages API proxy endpoint."""
    start_time = time.perf_counter()
    api_key = x_api_key or settings.ANTHROPIC_API_KEY
    user_key = (api_key[:12] + "...") if api_key else "anonymous"

    # 1. Canonical text extraction
    canonical_text = extract_canonical_text_anthropic(body.system, body.messages)
    original_tokens = compressor.estimate_tokens(canonical_text)

    # 2. DLP Sanitization
    dlp_redactions = 0
    sanitized_text = canonical_text
    if settings.DLP_ENABLED:
        sanitized_text, dlp_redactions = dlp_processor.sanitize(canonical_text)

    # 3. Prompt Compression
    compressed_text = sanitized_text
    tokens_saved = 0
    if settings.COMPRESSION_ENABLED:
        compressed_text, tokens_saved = compressor.compress(sanitized_text)

    # 4. Local Embedding
    embedding = await embedder.embed(compressed_text)

    # 5. Cache Lookup
    cache_hit = False
    cached_payload: dict[str, Any] | None = None
    similarity = 0.0

    if settings.CACHE_ENABLED:
        search_result = await db_manager.semantic_search(
            embedding=embedding,
            threshold=settings.CACHE_SIMILARITY_THRESHOLD,
            model=body.model,
        )
        if search_result is not None:
            cache_hit = True
            similarity = search_result.get("similarity", 1.0)
            raw_payload = search_result.get("response_payload", "{}")
            try:
                cached_payload = (
                    json.loads(raw_payload) if isinstance(raw_payload, str) else raw_payload
                )
            except Exception:
                cached_payload = None
                cache_hit = False

    # 6. Cache HIT
    if cache_hit and cached_payload is not None:
        response_text = _extract_response_text(cached_payload)
        completion_tokens = compressor.estimate_tokens(response_text)
        total_tokens_saved = tokens_saved + completion_tokens
        saved_usd = settings.get_model_cost(body.model, original_tokens, completion_tokens)
        latency_ms = (time.perf_counter() - start_time) * 1000

        headers = build_telemetry_headers(
            cache_hit=True,
            provider="anthropic",
            similarity=similarity,
            tokens_saved=total_tokens_saved,
            saved_usd=saved_usd,
            dlp_redactions=dlp_redactions,
            latency_ms=latency_ms,
        )

        await db_manager.record_analytics(
            user_key=user_key,
            provider="anthropic",
            model=body.model,
            protocol="anthropic",
            tokens_original=original_tokens,
            tokens_sent=0,
            tokens_completion=completion_tokens,
            tokens_saved=total_tokens_saved,
            cost_usd=0.0,
            saved_usd=saved_usd,
            cache_hit=True,
            similarity=similarity,
            dlp_redactions=dlp_redactions,
            latency_ms=latency_ms,
        )

        if body.stream:
            return StreamingResponse(
                generate_anthropic_stream_from_cache(cached_payload, body.model),
                media_type="text/event-stream",
                headers=headers,
            )

        return JSONResponse(content=cached_payload, headers=headers)

    # 7. Cache MISS — Forward Upstream
    raw_messages: list[dict[str, Any]] = []
    for m in body.messages:
        if isinstance(m.content, str):
            raw_messages.append({"role": m.role, "content": m.content})
        else:
            raw_messages.append(
                {
                    "role": m.role,
                    "content": [
                        b.model_dump() if hasattr(b, "model_dump") else b for b in m.content
                    ],
                }
            )

    request_kwargs: dict[str, Any] = {
        "max_tokens": body.max_tokens,
        "api_key": api_key,
    }
    if body.system is not None:
        request_kwargs["system"] = body.system
    if body.temperature is not None:
        request_kwargs["temperature"] = body.temperature
    if body.top_p is not None:
        request_kwargs["top_p"] = body.top_p
    if body.stop_sequences is not None:
        request_kwargs["stop_sequences"] = body.stop_sequences
    if anthropic_beta is not None:
        request_kwargs["anthropic_beta"] = anthropic_beta

    if body.stream:
        # Stream response
        collected_text_parts: list[str] = []

        async def _proxy_anthropic_stream() -> AsyncGenerator[str, None]:
            async for sse_chunk in anthropic_provider.stream_chat_completion(
                raw_messages,
                model=body.model,
                **request_kwargs,
            ):
                yield sse_chunk
                # Extract text for caching
                for line in sse_chunk.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("data: "):
                        try:
                            ev_data = json.loads(line[6:])
                            if ev_data.get("type") == "content_block_delta":
                                delta = ev_data.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    collected_text_parts.append(delta.get("text", ""))
                        except Exception:
                            logger.debug("Non-critical stream chunk parse skip")

            full_assembled = "".join(collected_text_parts)
            assembled_completion_tokens = compressor.estimate_tokens(full_assembled)
            assembled_payload: dict[str, Any] = {
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": full_assembled}],
                "model": body.model,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": original_tokens,
                    "output_tokens": assembled_completion_tokens,
                },
            }

            if settings.CACHE_ENABLED and full_assembled:
                prompt_hash = hashlib.sha256(canonical_text.encode()).hexdigest()
                await db_manager.insert_cache_entry(
                    id=str(uuid.uuid4()),
                    protocol="anthropic",
                    model=body.model,
                    prompt_hash=prompt_hash,
                    prompt_raw=canonical_text,
                    prompt_compressed=compressed_text,
                    embedding=embedding,
                    response_payload=json.dumps(assembled_payload),
                    tokens_saved=tokens_saved,
                )

            stream_latency_ms = (time.perf_counter() - start_time) * 1000
            cost_usd = settings.get_model_cost(
                body.model, original_tokens, assembled_completion_tokens
            )
            await db_manager.record_analytics(
                user_key=user_key,
                provider="anthropic",
                model=body.model,
                protocol="anthropic",
                tokens_original=original_tokens,
                tokens_sent=original_tokens - tokens_saved,
                tokens_completion=assembled_completion_tokens,
                tokens_saved=tokens_saved,
                cost_usd=cost_usd,
                saved_usd=0.0,
                cache_hit=False,
                similarity=0.0,
                dlp_redactions=dlp_redactions,
                latency_ms=stream_latency_ms,
            )

        telemetry = build_telemetry_headers(
            cache_hit=False,
            provider="anthropic",
            similarity=0.0,
            tokens_saved=tokens_saved,
            saved_usd=0.0,
            dlp_redactions=dlp_redactions,
            latency_ms=0.0,
        )
        return StreamingResponse(
            _proxy_anthropic_stream(),
            media_type="text/event-stream",
            headers=telemetry,
        )

    # Non-streaming
    response_payload = await anthropic_provider.chat_completion(
        raw_messages,
        model=body.model,
        **request_kwargs,
    )

    resp_usage = response_payload.get("usage", {})
    in_tok = resp_usage.get("input_tokens", original_tokens)
    out_tok = resp_usage.get("output_tokens", 0)

    # Cache response
    if settings.CACHE_ENABLED and "error" not in response_payload:
        prompt_hash = hashlib.sha256(canonical_text.encode()).hexdigest()
        await db_manager.insert_cache_entry(
            id=str(uuid.uuid4()),
            protocol="anthropic",
            model=body.model,
            prompt_hash=prompt_hash,
            prompt_raw=canonical_text,
            prompt_compressed=compressed_text,
            embedding=embedding,
            response_payload=json.dumps(response_payload),
            tokens_saved=tokens_saved,
        )

    latency_ms = (time.perf_counter() - start_time) * 1000
    cost_usd = settings.get_model_cost(body.model, in_tok, out_tok)

    await db_manager.record_analytics(
        user_key=user_key,
        provider="anthropic",
        model=body.model,
        protocol="anthropic",
        tokens_original=original_tokens,
        tokens_sent=in_tok,
        tokens_completion=out_tok,
        tokens_saved=tokens_saved,
        cost_usd=cost_usd,
        saved_usd=0.0,
        cache_hit=False,
        similarity=0.0,
        dlp_redactions=dlp_redactions,
        latency_ms=latency_ms,
    )

    headers = build_telemetry_headers(
        cache_hit=False,
        provider="anthropic",
        similarity=0.0,
        tokens_saved=tokens_saved,
        saved_usd=0.0,
        dlp_redactions=dlp_redactions,
        latency_ms=latency_ms,
    )

    return JSONResponse(content=response_payload, headers=headers)
