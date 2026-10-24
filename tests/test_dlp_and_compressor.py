from __future__ import annotations

from core.pipeline.compressor import compressor
from core.pipeline.dlp import dlp_processor

# ═══════════════════════════════════════════════════════════════════════════
#  DLP Processor — Redaction Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDLPRedaction:
    """Verify that sensitive data patterns are correctly redacted."""

    def test_redact_cpf(self) -> None:
        """CPF with dots/dash formatting is redacted."""
        text = "CPF: 123.456.789-00"
        result, count = dlp_processor.redact(text)
        assert "<REDACTED_CPF>" in result
        assert "123.456.789-00" not in result

    def test_redact_cpf_no_dots(self) -> None:
        """CPF without formatting (11 raw digits) is redacted."""
        text = "12345678900"
        result, count = dlp_processor.redact(text)
        assert "<REDACTED_CPF>" in result
        assert "12345678900" not in result

    def test_redact_cnpj(self) -> None:
        """CNPJ with standard formatting is redacted."""
        text = "CNPJ: 12.345.678/0001-90"
        result, count = dlp_processor.redact(text)
        assert "<REDACTED_CNPJ>" in result
        assert "12.345.678/0001-90" not in result

    def test_redact_email(self) -> None:
        """Email addresses are redacted."""
        text = "contact me at user@example.com"
        result, count = dlp_processor.redact(text)
        assert "<REDACTED_EMAIL>" in result
        assert "user@example.com" not in result

    def test_redact_credit_card(self) -> None:
        """Credit card numbers (dash-separated) are redacted."""
        text = "card 4111-1111-1111-1111"
        result, count = dlp_processor.redact(text)
        assert "<REDACTED_CC>" in result
        assert "4111-1111-1111-1111" not in result

    def test_redact_openai_key(self) -> None:
        """OpenAI API keys (sk-…) are redacted."""
        text = "key: sk-abc123def456ghi789jkl012"
        result, count = dlp_processor.redact(text)
        assert "<REDACTED_API_KEY>" in result
        assert "sk-abc123def456ghi789jkl012" not in result

    def test_redact_anthropic_key(self) -> None:
        """Anthropic API keys are redacted with the specific tag.

        Because the Anthropic pattern ``sk-ant-…`` is more specific than the
        generic ``sk-…`` pattern, it should be tagged ``<REDACTED_ANTHROPIC_KEY>``
        and NOT ``<REDACTED_API_KEY>``.
        """
        text = "key: sk-ant-abc123def456ghi789jkl012mno345pqr"
        result, count = dlp_processor.redact(text)
        assert "<REDACTED_ANTHROPIC_KEY>" in result
        assert "<REDACTED_API_KEY>" not in result
        assert "sk-ant-abc123def456ghi789jkl012mno345pqr" not in result

    def test_redact_github_token(self) -> None:
        """GitHub personal access tokens (ghp_…) are redacted."""
        text = "token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn"
        result, count = dlp_processor.redact(text)
        assert "<REDACTED_GH_TOKEN>" in result
        assert "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn" not in result

    def test_redact_aws_key(self) -> None:
        """AWS access key IDs (AKIA…) are redacted."""
        text = "aws: AKIAIOSFODNN7EXAMPLE"
        result, count = dlp_processor.redact(text)
        assert "<REDACTED_AWS_KEY>" in result
        assert "AKIAIOSFODNN7EXAMPLE" not in result

    def test_redact_count(self) -> None:
        """Redaction count matches the number of distinct sensitive items found."""
        text = "CPF: 123.456.789-00, email: admin@corp.io, card: 4111-1111-1111-1111"
        result, count = dlp_processor.redact(text)
        assert count == 3

    def test_no_false_positives(self) -> None:
        """Innocuous text passes through unchanged with zero redactions."""
        text = "The quick brown fox jumps over the lazy dog."
        result, count = dlp_processor.redact(text)
        assert result == text
        assert count == 0

    def test_detect_returns_details(self) -> None:
        """``detect()`` returns match details including type and value."""
        text = "CPF: 123.456.789-00 email: test@test.com"
        matches = dlp_processor.detect(text)
        assert len(matches) >= 2

        match_types = {m.pattern_name for m in matches}
        assert "CPF" in match_types or "cpf" in match_types.union({t.lower() for t in match_types})


# ═══════════════════════════════════════════════════════════════════════════
#  Compressor Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCompressor:
    """Verify prompt compression logic."""

    def test_collapse_multiple_spaces(self) -> None:
        """Multiple consecutive spaces are collapsed to a single space."""
        result = compressor.compress("hello    world")
        assert result.text == "hello world"

    def test_collapse_newlines(self) -> None:
        """Runs of 3+ newlines are reduced to a double newline."""
        result = compressor.compress("a\n\n\n\nb")
        assert result.text == "a\n\nb"

    def test_minify_json_block(self) -> None:
        """JSON blocks embedded in text are minified (whitespace removed)."""
        text = 'Here is data:\n```json\n{\n    "key":   "value",\n    "num": 42\n}\n```\nEnd.'
        result = compressor.compress(text)
        # The minified JSON should not contain the extra indentation.
        assert '    "key"' not in result.text
        # But the JSON values should still be present.
        assert '"key"' in result.text
        assert '"value"' in result.text

    def test_tokens_saved_positive(self) -> None:
        """Compressing redundant text reports a positive token savings."""
        bloated = "hello     world   \n\n\n\n  foo    bar"
        result = compressor.compress(bloated)
        assert result.tokens_saved > 0

    def test_estimate_tokens(self) -> None:
        """``estimate_tokens`` returns a reasonable positive integer."""
        token_count = compressor.estimate_tokens("Hello, world!")
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_idempotent(self) -> None:
        """Compressing already-compressed text is a no-op (0 tokens saved)."""
        clean = "hello world"
        first = compressor.compress(clean)
        second = compressor.compress(first.text)
        assert second.text == first.text
        assert second.tokens_saved == 0
