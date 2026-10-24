"""Heuristic Prompt Compressor module.

Applies lightweight, deterministic text transformations to reduce token
consumption before forwarding prompts upstream.  No model calls are made;
all savings come from whitespace normalisation and JSON minification.
"""

from __future__ import annotations

import json
import re
from re import Match
from typing import NamedTuple


class CompressionResult(NamedTuple):
    """Result of a prompt compression run."""

    text: str
    tokens_saved: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_FENCED_JSON_RE = re.compile(
    r"```json\s*\n(.*?)```",
    re.DOTALL,
)

_RAW_JSON_OBJECT_RE = re.compile(
    r"(?<![\"\\])(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})",
    re.DOTALL,
)

_RAW_JSON_ARRAY_RE = re.compile(
    r"(?<![\"\\])(\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])",
    re.DOTALL,
)


def _minify_json_match(match: Match[str]) -> str:
    """Attempt to parse and re-serialise the matched text as compact JSON.

    If the content is not valid JSON the original text is returned unchanged.
    """
    raw = match.group(1) if match.lastindex else match.group(0)
    try:
        parsed = json.loads(raw)
        minified = json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)
        # If the match came from a fenced block, keep the fence markers.
        if match.lastindex:
            return "```json\n" + minified + "```"
        return minified
    except (json.JSONDecodeError, ValueError, TypeError):
        return match.group(0)


# ---------------------------------------------------------------------------
# Compressor class
# ---------------------------------------------------------------------------


class PromptCompressor:
    """Apply heuristic compression to prompt text."""

    def __init__(self) -> None:
        pass  # No state required; kept for API symmetry.

    # -- public API ---------------------------------------------------------

    def compress(self, text: str) -> CompressionResult:
        """Compress *text* using whitespace and JSON heuristics.

        Returns:
            A ``CompressionResult(text, tokens_saved)`` named tuple.
        """
        original_len = len(text)

        # 1. Strip leading/trailing whitespace
        result = text.strip()

        # 2. Collapse multiple spaces → single space
        result = re.sub(r" {2,}", " ", result)

        # 3. Collapse 3+ newlines → double newline
        result = re.sub(r"\n{3,}", "\n\n", result)

        # 4. Minify fenced JSON blocks (```json … ```)
        result = _FENCED_JSON_RE.sub(_minify_json_match, result)

        # 4b. Minify inline/raw JSON objects and arrays
        result = _RAW_JSON_OBJECT_RE.sub(_minify_json_match, result)
        result = _RAW_JSON_ARRAY_RE.sub(_minify_json_match, result)

        # 5. Remove trailing whitespace from each line
        result = re.sub(r"[ \t]+$", "", result, flags=re.MULTILINE)

        # 6. Estimate tokens saved (rough 4-char-per-token heuristic)
        tokens_saved = max(0, (original_len - len(result)) // 4)

        return CompressionResult(text=result, tokens_saved=tokens_saved)

    def estimate_tokens(self, text: str) -> int:
        """Return a rough token-count estimate for *text*.

        Uses the simple heuristic of ~4 characters per token.
        """
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
compressor = PromptCompressor()
