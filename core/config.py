from __future__ import annotations

"""TokenMesh configuration — Pydantic Settings with pricing, currency helpers."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the TokenMesh gateway.

    Values are read from environment variables prefixed with ``TOKENMESH_``
    and/or a ``.env`` file located in the project root.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="TOKENMESH_",
        case_sensitive=False,
    )

    # ── Provider API keys ────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""

    # ── Server ───────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8787
    LOG_LEVEL: str = "info"

    # ── Storage ──────────────────────────────────────────────────────────
    DUCKDB_PATH: str = "data/tokenmesh.duckdb"

    # ── Semantic cache ───────────────────────────────────────────────────
    CACHE_SIMILARITY_THRESHOLD: float = 0.95
    CACHE_ENABLED: bool = True

    # ── DLP / compression feature flags ──────────────────────────────────
    DLP_ENABLED: bool = True
    COMPRESSION_ENABLED: bool = True
    MOCK_MODE: bool = False

    # ── CORS ─────────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["*"]

    # ── Embeddings ───────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM: int = 384

    # ── FinOps defaults ──────────────────────────────────────────────────
    DEFAULT_CURRENCY: str = "USD"
    DEFAULT_LANGUAGE: str = "pt-br"

    # ── Pricing table (USD per 1 M tokens) ───────────────────────────────
    PRICING_TABLE: dict[str, dict[str, float]] = Field(
        default={
            # OpenAI models
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-4": {"input": 30.00, "output": 60.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
            "o1": {"input": 15.00, "output": 60.00},
            "o1-mini": {"input": 3.00, "output": 12.00},
            "o3-mini": {"input": 1.10, "output": 4.40},
            # Anthropic models
            "claude-3-5-sonnet-latest": {"input": 3.00, "output": 15.00},
            "claude-3-7-sonnet-latest": {"input": 3.00, "output": 15.00},
            "claude-3-5-haiku-latest": {"input": 0.80, "output": 4.00},
            "claude-3-opus-latest": {"input": 15.00, "output": 75.00},
            "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
            "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        },
    )

    # ── Exchange rates (1 USD → X) ───────────────────────────────────────
    EXCHANGE_RATES: dict[str, float] = Field(
        default={
            "USD": 1.0,
            "BRL": 5.45,
            "EUR": 0.92,
            "GBP": 0.79,
            "JPY": 149.5,
            "CAD": 1.36,
            "CHF": 0.88,
            "AUD": 1.53,
            "MXN": 17.15,
            "ARS": 350.0,
        },
    )

    # ── Currency symbols ─────────────────────────────────────────────────
    CURRENCY_SYMBOLS: dict[str, str] = Field(
        default={
            "USD": "$",
            "BRL": "R$",
            "EUR": "€",
            "GBP": "£",
            "JPY": "¥",
            "CAD": "CA$",
            "CHF": "CHF",
            "AUD": "A$",
            "MXN": "MX$",
            "ARS": "AR$",
        },
    )

    # ── Helpers ──────────────────────────────────────────────────────────

    def get_model_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Return the cost **in USD** for a given model call.

        If the *model* is not present in the pricing table a default rate of
        $1.00 per 1 M tokens (both input and output) is used.
        """
        default_rate: dict[str, float] = {"input": 1.0, "output": 1.0}
        rates = self.PRICING_TABLE.get(model, default_rate)
        input_cost = (input_tokens / 1_000_000) * rates["input"]
        output_cost = (output_tokens / 1_000_000) * rates["output"]
        return input_cost + output_cost

    def convert_currency(
        self,
        usd_amount: float,
        target_currency: str,
        custom_rate: float | None = None,
    ) -> float:
        """Convert a USD amount to *target_currency*.

        When *custom_rate* is provided it takes precedence over the built-in
        exchange-rate table.  Falls back to 1.0 if the currency is unknown.
        """
        if custom_rate is not None:
            return usd_amount * custom_rate
        rate = self.EXCHANGE_RATES.get(target_currency.upper(), 1.0)
        return usd_amount * rate


settings = Settings()
