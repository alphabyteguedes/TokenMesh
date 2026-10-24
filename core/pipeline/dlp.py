"""Local Zero-Trust DLP (Data Loss Prevention) module.

Provides real-time pattern-based detection and redaction of sensitive data
including PII (CPF, CNPJ, email, credit cards) and secrets (API keys, tokens).
All processing happens locally — no data leaves the gateway.
"""

from __future__ import annotations

import re
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Pattern definition
# ---------------------------------------------------------------------------


class DLPPattern(NamedTuple):
    """A single DLP detection pattern."""

    name: str
    pattern: re.Pattern[str]
    replacement: str


class DLPMatch(NamedTuple):
    """A detected sensitive data occurrence."""

    pattern_name: str
    match: str
    start: int
    end: int


# Order matters: more specific patterns MUST come before less specific ones.
# e.g. ANTHROPIC_KEY (`sk-ant-…`) before OPENAI_KEY (`sk-…`).
DLP_PATTERNS: list[DLPPattern] = [
    DLPPattern(
        name="ANTHROPIC_KEY",
        pattern=re.compile(r"\bsk-ant-[a-zA-Z0-9_-]{20,}\b"),
        replacement="<REDACTED_ANTHROPIC_KEY>",
    ),
    DLPPattern(
        name="OPENAI_KEY",
        pattern=re.compile(r"\bsk-[a-zA-Z0-9_-]{20,}\b"),
        replacement="<REDACTED_API_KEY>",
    ),
    DLPPattern(
        name="GITHUB_TOKEN",
        pattern=re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{36,}\b"),
        replacement="<REDACTED_GH_TOKEN>",
    ),
    DLPPattern(
        name="AWS_KEY",
        pattern=re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        replacement="<REDACTED_AWS_KEY>",
    ),
    DLPPattern(
        name="CNPJ",
        pattern=re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b"),
        replacement="<REDACTED_CNPJ>",
    ),
    DLPPattern(
        name="CPF",
        pattern=re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
        replacement="<REDACTED_CPF>",
    ),
    DLPPattern(
        name="EMAIL",
        pattern=re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        replacement="<REDACTED_EMAIL>",
    ),
    DLPPattern(
        name="CREDIT_CARD",
        pattern=re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
        replacement="<REDACTED_CC>",
    ),
]

# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class DLPProcessor:
    """Applies DLP patterns to sanitize text and detect sensitive data."""

    def __init__(self) -> None:
        self._patterns: list[DLPPattern] = list(DLP_PATTERNS)

    # -- public API ---------------------------------------------------------

    def sanitize(self, text: str) -> tuple[str, int]:
        """Redact all sensitive patterns from *text*.

        Returns:
            A ``(sanitized_text, total_redaction_count)`` tuple where
            *total_redaction_count* is the cumulative number of individual
            matches replaced across **all** patterns.
        """
        total_count = 0
        sanitized = text
        for dlp_pattern in self._patterns:
            sanitized, n = dlp_pattern.pattern.subn(dlp_pattern.replacement, sanitized)
            total_count += n
        return sanitized, total_count

    def redact(self, text: str) -> tuple[str, int]:
        """Alias for :meth:`sanitize`."""
        return self.sanitize(text)

    def detect(self, text: str) -> list[DLPMatch]:
        """Return all sensitive-data matches found in *text* (for audit logging).

        Each entry contains the pattern name, the matched substring, and the
        ``start`` / ``end`` character offsets.  This method does **not** modify
        the text.
        """
        findings: list[DLPMatch] = []
        for dlp_pattern in self._patterns:
            for match in dlp_pattern.pattern.finditer(text):
                findings.append(
                    DLPMatch(
                        pattern_name=dlp_pattern.name,
                        match=match.group(),
                        start=match.start(),
                        end=match.end(),
                    )
                )
        return findings


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
dlp_processor = DLPProcessor()
