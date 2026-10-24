from __future__ import annotations

"""DuckDB-backed storage engine for semantic cache and analytics.

Provides async wrappers around synchronous DuckDB operations using
``asyncio.to_thread`` and an ``asyncio.Lock`` for write serialization.
"""

import asyncio
import datetime as dt
import logging
import uuid
from typing import Any

import duckdb

from core.config import settings

logger = logging.getLogger(__name__)


class CacheSearchResult(dict[str, Any]):
    """Dict subclass allowing dot notation for search result fields."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            if name == "response_text":
                return self.get("response_payload")
            raise AttributeError(f"'CacheSearchResult' object has no attribute '{name}'") from None


class AnalyticsSummaryResult(dict[str, Any]):
    """Dict subclass allowing dot notation for summary fields."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            if name == "cache_hit_count":
                return self.get("cache_hits", 0)
            if name == "total_tokens":
                return self.get(
                    "total_tokens_original",
                    self.get("total_tokens_sent", 0) + self.get("total_tokens_saved", 0),
                )
            raise AttributeError(
                f"'AnalyticsSummaryResult' object has no attribute '{name}'"
            ) from None


class DuckDBManager:
    """Manages a single DuckDB connection for the TokenMesh gateway.

    All public methods are async-safe. Write operations are serialized
    through ``_write_lock`` to avoid DuckDB's single-writer constraint.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._write_lock = asyncio.Lock()

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Open the DuckDB database and create tables / indexes."""
        self._conn = duckdb.connect(self.db_path)

        # Load the Vector Similarity Search extension
        self._conn.execute("INSTALL vss;")
        self._conn.execute("LOAD vss;")
        self._conn.execute("SET hnsw_enable_experimental_persistence = true;")

        # ── Semantic cache table ─────────────────────────────────────────
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_cache (
                id              VARCHAR PRIMARY KEY,
                protocol        VARCHAR NOT NULL,
                model           VARCHAR NOT NULL,
                prompt_hash     VARCHAR NOT NULL,
                prompt_raw      VARCHAR NOT NULL,
                prompt_compressed VARCHAR,
                embedding       FLOAT[384] NOT NULL,
                response_payload VARCHAR NOT NULL,
                tokens_saved    INTEGER NOT NULL DEFAULT 0,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        # ── Analytics events table ───────────────────────────────────────
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analytics_events (
                id                VARCHAR PRIMARY KEY,
                timestamp         TIMESTAMP NOT NULL,
                user_key          VARCHAR NOT NULL,
                provider          VARCHAR NOT NULL,
                model             VARCHAR NOT NULL,
                protocol          VARCHAR NOT NULL,
                tokens_original   INTEGER NOT NULL,
                tokens_sent       INTEGER NOT NULL,
                tokens_completion INTEGER NOT NULL,
                tokens_saved      INTEGER NOT NULL,
                cost_usd          DOUBLE NOT NULL,
                saved_usd         DOUBLE NOT NULL,
                cache_hit         BOOLEAN NOT NULL,
                similarity        DOUBLE NOT NULL DEFAULT 0.0,
                dlp_redactions    INTEGER NOT NULL DEFAULT 0,
                latency_ms        DOUBLE NOT NULL DEFAULT 0.0
            );
            """
        )

        # ── HNSW index (best-effort) ────────────────────────────────────
        try:
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_cache_embedding
                ON semantic_cache
                USING HNSW (embedding)
                WITH (metric = 'cosine');
                """
            )
        except Exception:
            logger.warning(
                "Could not create HNSW index on semantic_cache.embedding. "
                "This is expected when the table is empty or the VSS "
                "extension has compatibility issues. Falling back to "
                "brute-force cosine search.",
                exc_info=True,
            )

        logger.info("DuckDB initialized at %s", self.db_path)

    async def close(self) -> None:
        """Close the underlying DuckDB connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.info("DuckDB connection closed.")

    async def list_tables(self) -> list[str]:
        """Return a list of table names present in the database."""
        return await asyncio.to_thread(self._sync_list_tables)

    def _sync_list_tables(self) -> list[str]:
        if self._conn is None:
            return []
        try:
            rows = self._conn.execute("SHOW TABLES").fetchall()
            return [r[0] for r in rows]
        except Exception:
            return []

    # ── Semantic cache ───────────────────────────────────────────────────

    async def search_cache(
        self,
        embedding: list[float],
        threshold: float,
        model: str | None = None,
    ) -> CacheSearchResult | None:
        """Find the best semantic-cache match above *threshold* (returns dot-accessible result)."""
        res = await self.semantic_search(embedding=embedding, threshold=threshold, model=model)
        if res is None:
            return None
        return CacheSearchResult(res)

    async def semantic_search(
        self,
        embedding: list[float],
        threshold: float,
        model: str | None = None,
    ) -> dict[str, Any] | None:
        """Find the best semantic-cache match above *threshold*.

        Returns a dict with cache-hit metadata or ``None`` when nothing
        matches.
        """
        return await asyncio.to_thread(self._sync_semantic_search, embedding, threshold, model)

    def _sync_semantic_search(
        self,
        embedding: list[float],
        threshold: float,
        model: str | None = None,
    ) -> dict[str, Any] | None:
        """Synchronous implementation of :meth:`semantic_search`."""
        if self._conn is None:
            logger.error("DuckDB connection is not initialized.")
            return None

        try:
            # Check if the table has any rows to avoid unnecessary work
            row_count = self._conn.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()
            if row_count is None or row_count[0] == 0:
                return None
        except Exception:
            logger.warning("Failed to count semantic_cache rows.", exc_info=True)
            return None

        query = """
            SELECT id, protocol, model, prompt_raw, response_payload, tokens_saved,
                   array_cosine_similarity(embedding, ?::FLOAT[384]) AS similarity
            FROM semantic_cache
            WHERE array_cosine_similarity(embedding, ?::FLOAT[384]) >= ?
        """
        params: list[Any] = [embedding, embedding, threshold]

        if model is not None:
            query += " AND model = ?"
            params.append(model)

        query += " ORDER BY similarity DESC LIMIT 1"

        try:
            result = self._conn.execute(query, params).fetchone()
        except Exception:
            logger.warning("Semantic search query failed.", exc_info=True)
            return None

        if result is None:
            return None

        return CacheSearchResult(
            {
                "id": result[0],
                "protocol": result[1],
                "model": result[2],
                "prompt_raw": result[3],
                "response_payload": result[4],
                "tokens_saved": result[5],
                "similarity": result[6],
            }
        )

    async def insert_cache_entry(
        self,
        entry: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Insert a new entry into the semantic cache."""
        data: dict[str, Any] = dict(entry or {})
        data.update(kwargs)
        if "id" not in data:
            data["id"] = str(uuid.uuid4())
        if "protocol" not in data:
            data["protocol"] = data.get("provider", "openai")
        if "model" not in data:
            data["model"] = "unknown"
        if "prompt_hash" not in data:
            data["prompt_hash"] = data.get("request_hash", "")
        if "prompt_raw" not in data:
            data["prompt_raw"] = data.get("prompt_text", "")
        if "response_payload" not in data:
            data["response_payload"] = data.get("response_text", "")
        if "tokens_saved" not in data:
            data["tokens_saved"] = data.get("tokens_used", 0)

        async with self._write_lock:
            await asyncio.to_thread(self._sync_insert_cache, data)

    async def store_cache_entry(self, *args: Any, **kwargs: Any) -> None:
        """Alias for insert_cache_entry."""
        if args and isinstance(args[0], dict):
            await self.insert_cache_entry(args[0], **kwargs)
        else:
            await self.insert_cache_entry(**kwargs)

    def _sync_insert_cache(self, entry: dict[str, Any]) -> None:
        """Synchronous implementation of :meth:`insert_cache_entry`."""
        if self._conn is None:
            logger.error("DuckDB connection is not initialized.")
            return

        self._conn.execute(
            """
            INSERT INTO semantic_cache
                (id, protocol, model, prompt_hash, prompt_raw,
                 prompt_compressed, embedding, response_payload, tokens_saved)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                entry["id"],
                entry["protocol"],
                entry["model"],
                entry["prompt_hash"],
                entry["prompt_raw"],
                entry.get("prompt_compressed"),
                entry["embedding"],
                entry["response_payload"],
                entry.get("tokens_saved", 0),
            ],
        )

    # ── Analytics ────────────────────────────────────────────────────────

    async def record_analytics(
        self,
        event: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Persist a single analytics event."""
        data: dict[str, Any] = dict(event or {})
        data.update(kwargs)
        if "id" not in data:
            data["id"] = str(uuid.uuid4())
        if "timestamp" not in data:
            data["timestamp"] = dt.datetime.now(dt.UTC)
        if "user_key" not in data:
            data["user_key"] = data.get("user", "anonymous")
        if "provider" not in data:
            data["provider"] = "openai"
        if "model" not in data:
            data["model"] = "unknown"
        if "protocol" not in data:
            data["protocol"] = data.get("provider", "openai")
        if "tokens_original" not in data:
            data["tokens_original"] = data.get("prompt_tokens", 0) + data.get(
                "completion_tokens", 0
            )
        if "tokens_sent" not in data:
            data["tokens_sent"] = data.get("prompt_tokens", data.get("tokens_original", 0))
        if "tokens_completion" not in data:
            data["tokens_completion"] = data.get("completion_tokens", 0)
        if "tokens_saved" not in data:
            data["tokens_saved"] = 0
        if "cost_usd" not in data:
            data["cost_usd"] = 0.0
        if "saved_usd" not in data:
            data["saved_usd"] = 0.0
        if "cache_hit" not in data:
            data["cache_hit"] = False

        async with self._write_lock:
            await asyncio.to_thread(self._sync_record_analytics, data)

    async def insert_analytics_event(self, *args: Any, **kwargs: Any) -> None:
        """Alias for record_analytics."""
        if args and isinstance(args[0], dict):
            await self.record_analytics(args[0], **kwargs)
        else:
            await self.record_analytics(**kwargs)

    def _sync_record_analytics(self, event: dict[str, Any]) -> None:
        """Synchronous implementation of :meth:`record_analytics`."""
        if self._conn is None:
            logger.error("DuckDB connection is not initialized.")
            return

        self._conn.execute(
            """
            INSERT INTO analytics_events
                (id, timestamp, user_key, provider, model, protocol,
                 tokens_original, tokens_sent, tokens_completion, tokens_saved,
                 cost_usd, saved_usd, cache_hit, similarity,
                 dlp_redactions, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                event["id"],
                event["timestamp"],
                event["user_key"],
                event["provider"],
                event["model"],
                event["protocol"],
                event["tokens_original"],
                event["tokens_sent"],
                event["tokens_completion"],
                event["tokens_saved"],
                event["cost_usd"],
                event["saved_usd"],
                event["cache_hit"],
                event.get("similarity", 0.0),
                event.get("dlp_redactions", 0),
                event.get("latency_ms", 0.0),
            ],
        )

    async def get_analytics_summary(
        self,
        currency: str = "USD",
        custom_rate: float | None = None,
    ) -> dict[str, Any]:
        """Aggregate analytics and return a summary dict.

        Monetary values are converted to the requested *currency* using
        either *custom_rate* or the built-in exchange rates from
        :pydata:`core.config.settings`.
        """
        return await asyncio.to_thread(self._sync_get_analytics_summary, currency, custom_rate)

    def _sync_get_analytics_summary(
        self,
        currency: str,
        custom_rate: float | None,
    ) -> dict[str, Any]:
        """Synchronous implementation of :meth:`get_analytics_summary`."""
        if self._conn is None:
            logger.error("DuckDB connection is not initialized.")
            return self._empty_summary(currency)

        # Determine exchange rate
        if custom_rate is not None:
            rate = custom_rate
        else:
            rate = settings.EXCHANGE_RATES.get(currency.upper(), 1.0)

        currency_symbol = settings.CURRENCY_SYMBOLS.get(currency.upper(), currency)

        # ── Totals ───────────────────────────────────────────────────────
        try:
            row = self._conn.execute(
                """
                SELECT
                    COUNT(*)                       AS total_requests,
                    COALESCE(SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END), 0)
                                                   AS cache_hits,
                    COALESCE(SUM(tokens_original), 0) AS total_tokens_original,
                    COALESCE(SUM(tokens_sent), 0)     AS total_tokens_sent,
                    COALESCE(SUM(tokens_saved), 0)    AS total_tokens_saved,
                    COALESCE(SUM(cost_usd), 0.0)      AS total_cost_usd,
                    COALESCE(SUM(saved_usd), 0.0)     AS total_saved_usd,
                    COALESCE(AVG(latency_ms), 0.0)    AS avg_latency_ms,
                    COALESCE(SUM(dlp_redactions), 0)  AS total_dlp_redactions
                FROM analytics_events
                """
            ).fetchone()
        except Exception:
            logger.warning("Analytics aggregation query failed.", exc_info=True)
            return self._empty_summary(currency)

        if row is None:
            return self._empty_summary(currency)

        (
            total_requests,
            cache_hits,
            total_tokens_original,
            total_tokens_sent,
            total_tokens_saved,
            total_cost_usd,
            total_saved_usd,
            avg_latency_ms,
            total_dlp_redactions,
        ) = row

        cache_hit_rate = (cache_hits / total_requests * 100.0) if total_requests > 0 else 0.0
        total_cost = total_cost_usd * rate
        total_saved = total_saved_usd * rate
        net_cost = total_cost - total_saved
        savings_percentage = (
            (total_saved_usd / total_cost_usd * 100.0) if total_cost_usd > 0 else 0.0
        )

        # ── Top 5 models by request count ────────────────────────────────
        try:
            top_models_rows = self._conn.execute(
                """
                SELECT
                    model,
                    COUNT(*)                      AS request_count,
                    COALESCE(SUM(cost_usd), 0.0)  AS cost_usd,
                    COALESCE(SUM(saved_usd), 0.0) AS saved_usd,
                    COALESCE(SUM(tokens_sent), 0)  AS tokens_sent
                FROM analytics_events
                GROUP BY model
                ORDER BY request_count DESC
                LIMIT 5
                """
            ).fetchall()
        except Exception:
            logger.warning("Top models query failed.", exc_info=True)
            top_models_rows = []

        top_models: list[dict[str, Any]] = [
            {
                "model": r[0],
                "request_count": r[1],
                "cost": r[2] * rate,
                "saved": r[3] * rate,
                "tokens_sent": r[4],
            }
            for r in top_models_rows
        ]

        # ── Daily breakdown (last 30 days) ───────────────────────────────
        try:
            daily_rows = self._conn.execute(
                """
                SELECT
                    CAST(timestamp AS DATE)        AS day,
                    COUNT(*)                       AS requests,
                    COALESCE(SUM(cost_usd), 0.0)   AS cost_usd,
                    COALESCE(SUM(saved_usd), 0.0)  AS saved_usd,
                    COALESCE(SUM(tokens_sent), 0)   AS tokens_sent,
                    COALESCE(SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END), 0)
                                                   AS cache_hits
                FROM analytics_events
                WHERE timestamp >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY day
                ORDER BY day
                """
            ).fetchall()
        except Exception:
            logger.warning("Daily breakdown query failed.", exc_info=True)
            daily_rows = []

        daily_breakdown: list[dict[str, Any]] = [
            {
                "date": str(r[0]),
                "requests": r[1],
                "cost": r[2] * rate,
                "saved": r[3] * rate,
                "tokens_sent": r[4],
                "cache_hits": r[5],
            }
            for r in daily_rows
        ]

        return AnalyticsSummaryResult(
            {
                "total_requests": total_requests,
                "cache_hits": cache_hits,
                "cache_hit_rate": cache_hit_rate,
                "total_tokens_original": total_tokens_original,
                "total_tokens_sent": total_tokens_sent,
                "total_tokens_saved": total_tokens_saved,
                "total_cost": total_cost,
                "total_saved": total_saved,
                "net_cost": net_cost,
                "currency": currency.upper(),
                "currency_symbol": currency_symbol,
                "savings_percentage": savings_percentage,
                "avg_latency_ms": avg_latency_ms,
                "total_dlp_redactions": total_dlp_redactions,
                "top_models": top_models,
                "daily_breakdown": daily_breakdown,
            }
        )

    async def get_recent_events(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return recent analytics events ordered by timestamp descending."""
        return await asyncio.to_thread(self._sync_get_recent_events, limit, offset)

    def _sync_get_recent_events(self, limit: int, offset: int) -> list[dict[str, Any]]:
        if self._conn is None:
            return []
        try:
            rows = self._conn.execute(
                """
                SELECT id, timestamp, user_key, provider, model, protocol,
                       tokens_original, tokens_sent, tokens_completion, tokens_saved,
                       cost_usd, saved_usd, cache_hit, similarity, dlp_redactions, latency_ms
                FROM analytics_events
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                [limit, offset],
            ).fetchall()
            return [
                {
                    "id": r[0],
                    "timestamp": r[1].isoformat() if hasattr(r[1], "isoformat") else str(r[1]),
                    "user_key": r[2],
                    "provider": r[3],
                    "model": r[4],
                    "protocol": r[5],
                    "tokens_original": r[6],
                    "tokens_sent": r[7],
                    "tokens_completion": r[8],
                    "tokens_saved": r[9],
                    "cost_usd": r[10],
                    "saved_usd": r[11],
                    "cache_hit": r[12],
                    "similarity": r[13],
                    "dlp_redactions": r[14],
                    "latency_ms": r[15],
                }
                for r in rows
            ]
        except Exception:
            logger.warning("Failed to fetch recent analytics events.", exc_info=True)
            return []

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _empty_summary(currency: str) -> AnalyticsSummaryResult:
        """Return a zeroed-out summary dict for the given currency."""
        currency_symbol = settings.CURRENCY_SYMBOLS.get(currency.upper(), currency)
        return AnalyticsSummaryResult(
            {
                "total_requests": 0,
                "cache_hits": 0,
                "cache_hit_rate": 0.0,
                "total_tokens_original": 0,
                "total_tokens_sent": 0,
                "total_tokens_saved": 0,
                "total_cost": 0.0,
                "total_saved": 0.0,
                "net_cost": 0.0,
                "currency": currency.upper(),
                "currency_symbol": currency_symbol,
                "savings_percentage": 0.0,
                "avg_latency_ms": 0.0,
                "total_dlp_redactions": 0,
                "top_models": [],
                "daily_breakdown": [],
            }
        )


db_manager = DuckDBManager(settings.DUCKDB_PATH)
