from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from core.config import settings
from core.db.duckdb_manager import db_manager
from core.pipeline.compressor import compressor
from core.pipeline.dlp import dlp_processor
from core.pipeline.embedder import embedder
from core.providers.anthropic_provider import AnthropicProvider, anthropic_provider
from core.providers.openai_provider import openai_provider
from core.schemas.openai_schema import OpenAIChatRequest

logger = logging.getLogger("tokenmesh.routes.openai")

router = APIRouter()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COST_PER_1K_TOKENS: float = 0.002  # rough average for estimation


def extract_canonical_text(messages: list[dict[str, Any]]) -> str:
    """Extract and concatenate text from OpenAI-format messages.

    Handles both simple ``{"role": ..., "content": "..."}`` and the
    multi-part ``{"role": ..., "content": [{"type": "text", "text": "..."}]}``
    format.  Only ``system`` and ``user`` messages are considered for the
    canonical representation used in embedding / caching.
    """
    parts: list[str] = []
    for msg in messages:
        role: str = msg.get("role", "")
        if role not in ("system", "user"):
            continue
        content = msg.get("content")
        if content is None:
            continue
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
    return "\n".join(parts)


async def generate_openai_stream_from_cache(
    cached_response: dict[str, Any],
) -> AsyncGenerator[str, None]:
    """Yield SSE events that reconstruct *cached_response* as an OpenAI stream.

    The cached response is expected to follow the ``/v1/chat/completions``
    JSON shape.  The assistant content is broken into small word-level chunks
    and emitted as ``chat.completion.chunk`` delta events.
    """
    response_id: str = cached_response.get("id", f"chatcmpl-{uuid.uuid4().hex[:29]}")
    model: str = cached_response.get("model", "unknown")
    created: int = cached_response.get("created", int(time.time()))

    # Gather full assistant text from choices
    full_text = ""
    for choice in cached_response.get("choices", []):
        message = choice.get("message", {})
        full_text += message.get("content", "")

    # Break into word-level chunks
    words = full_text.split(" ")
    chunks: list[str] = []
    for i, word in enumerate(words):
        chunks.append(word if i == 0 else f" {word}")

    # Emit role delta first
    role_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(role_chunk)}\n\n"

    # Emit content deltas
    for chunk_text in chunks:
        delta_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk_text},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(delta_chunk)}\n\n"

    # Emit finish
    finish_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(finish_chunk)}\n\n"
    yield "data: [DONE]\n\n"


def build_telemetry_headers(
    *,
    cache_hit: bool,
    provider: str,
    similarity: float,
    tokens_saved: int,
    saved_usd: float,
    dlp_redactions: int,
    latency_ms: float,
) -> dict[str, str]:
    """Build the ``X-TokenMesh-*`` telemetry headers."""
    return {
        "X-TokenMesh-Cache": "HIT" if cache_hit else "MISS",
        "X-TokenMesh-Provider": provider,
        "X-TokenMesh-Similarity": f"{similarity:.4f}",
        "X-TokenMesh-Tokens-Saved": str(tokens_saved),
        "X-TokenMesh-Saved-USD": f"{saved_usd:.6f}",
        "X-TokenMesh-DLP-Redactions": str(dlp_redactions),
        "X-TokenMesh-Latency-Ms": f"{latency_ms:.1f}",
    }


def _estimate_tokens(text: str) -> int:
    """Rough token count estimation (~4 chars per token)."""
    return max(1, len(text) // 4)


def _estimate_savings_usd(tokens_saved: int) -> float:
    """Estimate USD savings based on token count."""
    return (tokens_saved / 1000) * COST_PER_1K_TOKENS


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------


@router.post("/v1/chat/completions")
async def chat_completions(body: OpenAIChatRequest, request: Request) -> Any:
    """OpenAI-compatible chat completions endpoint — TokenMesh gateway."""
    start_time = time.perf_counter()

    # -- Auth -----------------------------------------------------------------
    auth_header: str = request.headers.get("authorization", "")
    api_key: str = auth_header.removeprefix("Bearer ").strip()

    # -- Client overrides (Live Playground support) ---------------------------
    dlp_header = request.headers.get("x-tokenmesh-dlp")
    comp_header = request.headers.get("x-tokenmesh-compression")
    thresh_header = request.headers.get("x-tokenmesh-threshold")
    inspect_header = request.headers.get("x-tokenmesh-inspect") or request.query_params.get(
        "inspect"
    )

    dlp_enabled = settings.DLP_ENABLED if dlp_header is None else dlp_header.lower() != "false"
    comp_enabled = (
        settings.COMPRESSION_ENABLED if comp_header is None else comp_header.lower() != "false"
    )
    threshold = (
        settings.CACHE_SIMILARITY_THRESHOLD if thresh_header is None else float(thresh_header)
    )

    # -- 1. Extract canonical text -------------------------------------------
    raw_messages: list[dict[str, Any]] = [m.model_dump() for m in body.messages]
    canonical_text: str = extract_canonical_text(raw_messages)

    # -- 2. DLP sanitization -------------------------------------------------
    dlp_redactions: int = 0
    sanitized_text: str = canonical_text
    if dlp_enabled:
        sanitized_text, dlp_redactions = dlp_processor.sanitize(canonical_text)

    # -- 3. Compression ------------------------------------------------------
    tokens_saved: int = 0
    compressed_text: str = sanitized_text
    if comp_enabled:
        compressed_text, tokens_saved = compressor.compress(sanitized_text)

    # -- 4. Embedding --------------------------------------------------------
    embedding: list[float] = await embedder.embed(compressed_text)

    # -- 5. Cache lookup -----------------------------------------------------
    cache_hit: bool = False
    cached_response: dict[str, Any] | None = None
    similarity: float = 0.0
    provider_name: str = "cache"

    if settings.CACHE_ENABLED:
        search_result = await db_manager.semantic_search(embedding, threshold)
        if search_result is not None:
            cache_hit = True
            similarity = search_result.get("similarity", 1.0)
            raw_payload = search_result.get("response_payload", "{}")
            try:
                cached_response = (
                    json.loads(raw_payload) if isinstance(raw_payload, str) else raw_payload
                )
            except Exception:
                cached_response = None
                cache_hit = False

    # -- 6. Cache HIT --------------------------------------------------------
    if cache_hit and cached_response is not None:
        # Estimate saved tokens from cached response
        cached_text = ""
        for choice in cached_response.get("choices", []):
            cached_text += choice.get("message", {}).get("content", "")
        total_tokens_saved = tokens_saved + _estimate_tokens(cached_text)
        saved_usd = _estimate_savings_usd(total_tokens_saved)
        latency_ms = (time.perf_counter() - start_time) * 1000

        headers = build_telemetry_headers(
            cache_hit=True,
            provider="cache",
            similarity=similarity,
            tokens_saved=total_tokens_saved,
            saved_usd=saved_usd,
            dlp_redactions=dlp_redactions,
            latency_ms=latency_ms,
        )

        # Record analytics
        await db_manager.record_analytics(
            model=body.model,
            provider="cache",
            cache_hit=True,
            tokens_saved=total_tokens_saved,
            saved_usd=saved_usd,
            latency_ms=latency_ms,
            dlp_redactions=dlp_redactions,
        )

        if body.stream:
            return StreamingResponse(
                generate_openai_stream_from_cache(cached_response),
                media_type="text/event-stream",
                headers=headers,
            )
        if inspect_header and isinstance(cached_response, dict):
            cached_response = dict(cached_response)
            cached_response["_tokenmesh"] = {
                "cache_hit": True,
                "similarity": similarity,
                "tokens_saved": total_tokens_saved,
                "saved_usd": saved_usd,
                "dlp_redactions": dlp_redactions,
                "latency_ms": latency_ms,
                "sanitized_text": sanitized_text,
                "compressed_text": compressed_text,
                "embedding_preview": [round(x, 4) for x in embedding[:6]],
            }
        return JSONResponse(content=cached_response, headers=headers)

    # -- 7. Cache MISS — forward to upstream provider ------------------------
    provider_name = "openai"
    response_payload: dict[str, Any]

    if body.model.startswith("claude-"):
        # Translate to Anthropic and back
        provider_name = "anthropic"
        anthropic_payload = AnthropicProvider.openai_to_anthropic(body.model_dump())
        anthropic_response = await anthropic_provider.chat_completion(
            anthropic_payload, api_key=api_key, stream=False
        )
        response_payload = AnthropicProvider.anthropic_to_openai_response(
            anthropic_response, model=body.model
        )
    else:
        if body.stream:
            # For streaming, we proxy the upstream stream and collect the full
            # response for caching after all chunks have been forwarded.
            collected_chunks: list[dict[str, Any]] = []
            full_content_parts: list[str] = []

            async def _proxy_and_collect() -> AsyncGenerator[str, None]:
                nonlocal response_payload
                async for chunk_bytes in openai_provider.stream_chat_completion(
                    body.model_dump(), api_key=api_key
                ):
                    chunk_str = (
                        chunk_bytes if isinstance(chunk_bytes, str) else chunk_bytes.decode()
                    )
                    yield chunk_str

                    # Try to extract content for caching
                    for line in chunk_str.strip().split("\n"):
                        line = line.strip()
                        if line.startswith("data: ") and line != "data: [DONE]":
                            try:
                                chunk_data = json.loads(line[6:])
                                collected_chunks.append(chunk_data)
                                for choice in chunk_data.get("choices", []):
                                    delta = choice.get("delta", {})
                                    content = delta.get("content")
                                    if content:
                                        full_content_parts.append(content)
                            except json.JSONDecodeError:
                                pass

                # After stream completes, build cacheable response and store it
                assembled_text = "".join(full_content_parts)
                assembled_response: dict[str, Any] = {
                    "id": collected_chunks[0].get("id", f"chatcmpl-{uuid.uuid4().hex[:29]}")
                    if collected_chunks
                    else f"chatcmpl-{uuid.uuid4().hex[:29]}",
                    "object": "chat.completion",
                    "created": collected_chunks[0].get("created", int(time.time()))
                    if collected_chunks
                    else int(time.time()),
                    "model": body.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": assembled_text,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": _estimate_tokens(canonical_text),
                        "completion_tokens": _estimate_tokens(assembled_text),
                        "total_tokens": _estimate_tokens(canonical_text)
                        + _estimate_tokens(assembled_text),
                    },
                }

                # Cache the assembled response
                if settings.CACHE_ENABLED and assembled_text:
                    request_hash = hashlib.sha256(canonical_text.encode()).hexdigest()
                    await db_manager.store_cache_entry(
                        id=str(uuid.uuid4()),
                        protocol="openai",
                        prompt_hash=request_hash,
                        prompt_raw=canonical_text,
                        prompt_compressed=compressed_text,
                        embedding=embedding,
                        response_payload=json.dumps(assembled_response),
                        model=body.model,
                        tokens_saved=tokens_saved,
                    )

                latency_ms_inner = (time.perf_counter() - start_time) * 1000
                await db_manager.record_analytics(
                    model=body.model,
                    provider=provider_name,
                    cache_hit=False,
                    tokens_saved=tokens_saved,
                    saved_usd=_estimate_savings_usd(tokens_saved),
                    latency_ms=latency_ms_inner,
                    dlp_redactions=dlp_redactions,
                )

            telemetry = build_telemetry_headers(
                cache_hit=False,
                provider=provider_name,
                similarity=0.0,
                tokens_saved=tokens_saved,
                saved_usd=_estimate_savings_usd(tokens_saved),
                dlp_redactions=dlp_redactions,
                latency_ms=0.0,  # not yet known for streaming
            )
            return StreamingResponse(
                _proxy_and_collect(),
                media_type="text/event-stream",
                headers=telemetry,
            )

        # Non-streaming OpenAI
        response_payload = await openai_provider.chat_completion(body.model_dump(), api_key=api_key)

    # -- Cache the response ---------------------------------------------------
    if settings.CACHE_ENABLED and "error" not in response_payload:
        request_hash = hashlib.sha256(canonical_text.encode()).hexdigest()
        await db_manager.store_cache_entry(
            id=str(uuid.uuid4()),
            protocol="openai",
            prompt_hash=request_hash,
            prompt_raw=canonical_text,
            prompt_compressed=compressed_text,
            embedding=embedding,
            response_payload=json.dumps(response_payload),
            model=body.model,
            tokens_saved=tokens_saved,
        )

    # -- 8. Analytics ---------------------------------------------------------
    latency_ms = (time.perf_counter() - start_time) * 1000
    saved_usd = _estimate_savings_usd(tokens_saved)

    await db_manager.record_analytics(
        model=body.model,
        provider=provider_name,
        cache_hit=False,
        tokens_saved=tokens_saved,
        saved_usd=saved_usd,
        latency_ms=latency_ms,
        dlp_redactions=dlp_redactions,
    )

    # -- 9. Telemetry headers -------------------------------------------------
    headers = build_telemetry_headers(
        cache_hit=False,
        provider=provider_name,
        similarity=0.0,
        tokens_saved=tokens_saved,
        saved_usd=saved_usd,
        dlp_redactions=dlp_redactions,
        latency_ms=latency_ms,
    )

    if inspect_header and isinstance(response_payload, dict):
        response_payload = dict(response_payload)
        response_payload["_tokenmesh"] = {
            "cache_hit": False,
            "similarity": 0.0,
            "tokens_saved": tokens_saved,
            "saved_usd": saved_usd,
            "dlp_redactions": dlp_redactions,
            "latency_ms": latency_ms,
            "sanitized_text": sanitized_text,
            "compressed_text": compressed_text,
            "embedding_preview": [round(x, 4) for x in embedding[:6]],
        }

    return JSONResponse(content=response_payload, headers=headers)
