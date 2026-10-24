from __future__ import annotations

import asyncio
import os
import tempfile
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio

from core.db.duckdb_manager import DuckDBManager

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 384
KNOWN_EMBEDDING: list[float] = [0.1] * EMBEDDING_DIM
DIFFERENT_EMBEDDING: list[float] = [-0.1] * (EMBEDDING_DIM // 2) + [0.1] * (EMBEDDING_DIM // 2)


# ---------------------------------------------------------------------------
# Fixtures (self-contained — does NOT rely on conftest.tmp_db so this
# module can be run in isolation).
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture
async def tmp_db() -> AsyncGenerator[DuckDBManager, None]:
    """Provide a fresh, initialised DuckDB instance for each test."""
    tmp = tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    db = DuckDBManager(db_path=tmp_path)
    await db.initialize()

    yield db

    await db.close()
    for suffix in ("", ".wal"):
        try:
            os.unlink(f"{tmp_path}{suffix}")
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════════════
#  Table Initialisation
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_initialize_creates_tables(tmp_db: DuckDBManager) -> None:
    """After ``initialize()``, the semantic_cache and analytics_events tables must exist."""
    tables = await tmp_db.list_tables()
    table_names = {t.lower() for t in tables}
    assert "semantic_cache" in table_names, f"Expected 'semantic_cache' in {table_names}"
    assert "analytics_events" in table_names, f"Expected 'analytics_events' in {table_names}"


# ═══════════════════════════════════════════════════════════════════════════
#  Semantic Cache — Insert & Search
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_insert_and_search_cache(tmp_db: DuckDBManager) -> None:
    """A cached entry can be retrieved by searching with its own embedding."""
    await tmp_db.insert_cache_entry(
        prompt_hash="hash_abc",
        prompt_text="What is the meaning of life?",
        response_text="42",
        embedding=KNOWN_EMBEDDING,
        model="gpt-4o",
        provider="openai",
        tokens_used=18,
        cost_usd=0.001,
    )

    hit = await tmp_db.search_cache(embedding=KNOWN_EMBEDDING, threshold=0.9)
    assert hit is not None
    assert hit.response_text == "42"
    assert hit.similarity >= 0.99  # same vector → cosine similarity ≈ 1.0


@pytest.mark.asyncio
async def test_search_below_threshold_returns_none(tmp_db: DuckDBManager) -> None:
    """Searching with a dissimilar embedding below threshold yields ``None``."""
    await tmp_db.insert_cache_entry(
        prompt_hash="hash_xyz",
        prompt_text="Hello world",
        response_text="Hi there",
        embedding=KNOWN_EMBEDDING,
        model="gpt-4o",
        provider="openai",
        tokens_used=10,
        cost_usd=0.0005,
    )

    hit = await tmp_db.search_cache(embedding=DIFFERENT_EMBEDDING, threshold=0.99)
    assert hit is None


# ═══════════════════════════════════════════════════════════════════════════
#  Analytics — Insert & Query
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_insert_and_query_analytics(tmp_db: DuckDBManager) -> None:
    """Inserted analytics events are reflected in the summary."""
    await tmp_db.insert_analytics_event(
        provider="openai",
        model="gpt-4o",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        cost_usd=0.005,
        latency_ms=320.0,
        cache_hit=False,
    )
    await tmp_db.insert_analytics_event(
        provider="openai",
        model="gpt-4o",
        prompt_tokens=80,
        completion_tokens=40,
        total_tokens=120,
        cost_usd=0.004,
        latency_ms=280.0,
        cache_hit=True,
    )

    summary = await tmp_db.get_analytics_summary(currency="USD")
    assert summary.total_requests == 2
    assert summary.total_tokens == 270
    assert summary.cache_hit_count == 1


@pytest.mark.asyncio
async def test_analytics_summary_currency_conversion(tmp_db: DuckDBManager) -> None:
    """When currency is BRL the amounts are converted from USD."""
    await tmp_db.insert_analytics_event(
        provider="anthropic",
        model="claude-3-5-sonnet-latest",
        prompt_tokens=50,
        completion_tokens=25,
        total_tokens=75,
        cost_usd=0.01,
        latency_ms=200.0,
        cache_hit=False,
    )

    summary_usd = await tmp_db.get_analytics_summary(currency="USD")
    summary_brl = await tmp_db.get_analytics_summary(currency="BRL")

    # BRL amounts must be strictly greater than USD (rate > 1).
    assert summary_brl.total_cost > summary_usd.total_cost


# ═══════════════════════════════════════════════════════════════════════════
#  Concurrency Safety
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_concurrent_writes(tmp_db: DuckDBManager) -> None:
    """Multiple concurrent inserts must complete without corruption."""

    async def _insert(index: int) -> None:
        # Each task uses a slightly perturbed embedding so rows differ.
        embedding = [0.1 + (index * 0.001)] * EMBEDDING_DIM
        await tmp_db.insert_cache_entry(
            prompt_hash=f"concurrent_{index}",
            prompt_text=f"question {index}",
            response_text=f"answer {index}",
            embedding=embedding,
            model="gpt-4o",
            provider="openai",
            tokens_used=10 + index,
            cost_usd=0.0001 * index,
        )

    tasks = [asyncio.create_task(_insert(i)) for i in range(20)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # No task should have raised.
    for i, result in enumerate(results):
        assert not isinstance(result, Exception), f"Concurrent write {i} failed: {result}"
