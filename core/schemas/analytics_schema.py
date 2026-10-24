from __future__ import annotations

"""Pydantic v2 schemas for FinOps analytics events and summaries."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class AnalyticsEvent(BaseModel):
    """A single analytics event recorded after every proxy request."""

    id: str
    timestamp: datetime
    user_key: str
    provider: str
    model: str
    protocol: str
    tokens_original: int
    tokens_sent: int
    tokens_completion: int
    tokens_saved: int
    cost_usd: float
    saved_usd: float
    cache_hit: bool
    similarity: float
    dlp_redactions: int
    latency_ms: float


class CurrencyConfig(BaseModel):
    """User-supplied currency preferences for analytics output."""

    currency: str = "USD"
    custom_currency_symbol: str | None = None
    custom_exchange_rate: float | None = None


class AnalyticsSummary(BaseModel):
    """Aggregated analytics summary returned by the dashboard API."""

    total_requests: int
    cache_hits: int
    cache_hit_rate: float
    total_tokens_original: int
    total_tokens_sent: int
    total_tokens_saved: int
    total_cost: float
    total_saved: float
    net_cost: float
    currency: str
    currency_symbol: str
    savings_percentage: float
    avg_latency_ms: float
    total_dlp_redactions: int
    top_models: list[dict[str, Any]]
    daily_breakdown: list[dict[str, Any]]
