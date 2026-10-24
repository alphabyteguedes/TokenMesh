from __future__ import annotations

"""FinOps analytics routes for real-time cost, savings, and telemetry reporting.

Supports dynamic multi-currency conversion with built-in presets (BRL, USD,
EUR, GBP, JPY, CAD, CHF, AUD, MXN, ARS) and arbitrary custom exchange rates.
Powers the Obsidian Linear.app dashboard and external FinOps integrations.
"""

import logging
from typing import Any

from fastapi import APIRouter, Query

from core.config import settings
from core.db.duckdb_manager import db_manager

logger = logging.getLogger("tokenmesh.routes.finops")

router = APIRouter()


@router.get("/v1/analytics/summary")
async def get_analytics_summary(
    currency: str = Query("USD", description="Currency code (e.g. USD, BRL, EUR)"),
    custom_currency_symbol: str | None = Query(
        None, description="Custom currency symbol (e.g. R$)"
    ),
    custom_symbol: str | None = Query(None, description="Alias for custom_currency_symbol"),
    custom_exchange_rate: float | None = Query(None, description="Custom USD exchange rate"),
    custom_rate: float | None = Query(None, description="Alias for custom_exchange_rate"),
) -> dict[str, Any]:
    """Retrieve aggregated FinOps analytics summary with currency conversion."""
    rate = custom_exchange_rate if custom_exchange_rate is not None else custom_rate
    symbol_override = custom_currency_symbol or custom_symbol

    curr_code = currency.upper()
    summary = await db_manager.get_analytics_summary(currency=curr_code, custom_rate=rate)

    if symbol_override:
        summary["currency_symbol"] = symbol_override

    # Provide frontend-friendly KPI and chart structures for the Linear.app dashboard
    summary["kpi"] = {
        "total_requests": summary.get("total_requests", 0),
        "requests_trend": 0,
        "cache_hit_rate": summary.get("cache_hit_rate", 0.0),
        "cache_trend": 0,
        "tokens_saved": summary.get("total_tokens_saved", 0),
        "tokens_trend": 0,
        "money_saved": summary.get("total_saved", 0.0),
        "money_trend": 0,
    }

    daily = summary.get("daily_breakdown", [])
    summary["daily_savings"] = [
        {"date": d.get("date", ""), "amount": d.get("saved", 0.0)} for d in daily
    ]

    recent_events = await db_manager.get_recent_events(limit=20, offset=0)
    summary["recent_activity"] = [
        {
            "timestamp": e.get("timestamp", ""),
            "provider": e.get("provider", ""),
            "model": e.get("model", ""),
            "tokens_original": e.get("tokens_original", 0),
            "tokens_sent": e.get("tokens_sent", 0),
            "cost": e.get("cost_usd", 0.0)
            * (rate if rate is not None else settings.EXCHANGE_RATES.get(curr_code, 1.0)),
            "cache_hit": e.get("cache_hit", False),
            "dlp_redactions": e.get("dlp_redactions", 0),
        }
        for e in recent_events
    ]

    return summary


@router.get("/v1/analytics/events")
async def get_analytics_events(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of events to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
) -> list[dict[str, Any]]:
    """Retrieve raw historical analytics events with pagination."""
    return await db_manager.get_recent_events(limit=limit, offset=offset)
