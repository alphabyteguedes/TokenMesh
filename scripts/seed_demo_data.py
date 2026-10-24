from __future__ import annotations

"""Seed realistic demo analytics data into TokenMesh DuckDB database.

Generates 150+ realistic historical analytics events across OpenAI and Anthropic
models distributed over the last 7 days, including DLP redactions, cache hits,
and multi-currency financial savings.

Usage:
    python scripts/seed_demo_data.py
"""

import asyncio
import datetime as dt
import random
import uuid

from core.config import settings
from core.db.duckdb_manager import db_manager

MODELS = [
    ("openai", "gpt-4o-mini", 0.00015, 0.0006),
    ("openai", "gpt-4o", 0.0025, 0.010),
    ("anthropic", "claude-3-5-sonnet-latest", 0.003, 0.015),
    ("anthropic", "claude-3-5-haiku-latest", 0.0008, 0.004),
]

USERS = ["developer_alice", "engineer_bob", "cursor_ide", "claude_code_cli", "ci_pipeline"]

SAMPLE_PROMPTS = [
    "How to configure Nginx as a reverse proxy with TLS 1.3?",
    "Explain the difference between optimistic and pessimistic locking in SQL.",
    "Implement a debounce function in TypeScript with cancel method.",
    "How does the Paxos algorithm guarantee safety in distributed systems?",
    "Sanitize sensitive customer CPF 123.456.789-00 and refund transaction.",
    "Explain the architecture of Transformer multi-head attention.",
    "Write a Dockerfile multi-stage build for a FastAPI Python application.",
    "How to index high-dimensional embeddings using HNSW graphs?",
]


async def seed_via_api(base_url: str) -> bool:
    """Seed data by sending requests to running TokenMesh server."""
    import httpx

    print(f"Server is running at {base_url}. Seeding through live HTTP API...")
    url = f"{base_url.rstrip('/')}/v1/chat/completions"

    test_prompts = [
        "What is the time complexity of QuickSort in worst case?",
        "Explain how DuckDB uses columnar vectorization for OLAP.",
        "Process customer CPF 456.789.123-99 for user_id alice@corp.internal",
        "Implement a binary search tree deletion in Python.",
        "What is the difference between TCP and UDP protocols?",
        "Sanitize CNPJ 12.345.678/0001-90 and AWS key AKIAIOSFODNN7EXAMPLE",
        "Explain the HNSW vector index construction algorithm in detail.",
        "How does semantic caching reduce LLM inference latency?",
    ]

    models = ["gpt-4o-mini", "claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"]

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Phase 1: Priming misses
        print("  -> Sending initial queries (Cache MISS & DLP sanitization)...")
        for prompt in test_prompts:
            model = random.choice(models)
            await client.post(
                url,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )

        # Phase 2: Generating cache hits
        print("  -> Repeating queries (Sub-5ms Cache HIT)...")
        for _ in range(25):
            prompt = random.choice(test_prompts)
            model = random.choice(models)
            await client.post(
                url,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )

    print("\n [OK] Live API seeding complete!")
    print(f" View updated metrics at: {base_url}/dashboard\n")
    return True


async def seed() -> None:
    print("=" * 60)
    print(" TOKENMESH DEMO DATA SEEDER")
    print(f" Target Database: {settings.DUCKDB_PATH}")
    print("=" * 60)

    # First check if the gateway server is already running on http://localhost:8787
    import logging

    import httpx

    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            resp = await client.get("http://localhost:8787/healthz")
            if resp.status_code == 200:
                await seed_via_api("http://localhost:8787")
                return
    except Exception as exc:
        logging.getLogger(__name__).debug("Gateway not reachable over HTTP: %s", exc)

    try:
        await db_manager.initialize()
    except Exception as exc:
        print(f"\n[!] Could not open DuckDB directly: {exc}")
        print("    If TokenMesh is running, make sure port 8787 is accessible.\n")
        return

    now = dt.datetime.now(dt.UTC)
    total_events = 160
    cache_hit_count = 0
    total_saved_usd = 0.0
    total_cost_usd = 0.0

    print(f"Generating {total_events} realistic events over the past 7 days...")

    for _i in range(total_events):
        # Distribute over the past 7 days with more events on recent days
        day_offset = random.choices([0, 1, 2, 3, 4, 5, 6], weights=[35, 25, 18, 10, 6, 4, 2])[0]
        minute_offset = random.randint(0, 1440)
        event_time = now - dt.timedelta(days=day_offset, minutes=minute_offset)

        provider, model, in_cost, out_cost = random.choice(MODELS)
        user_key = random.choice(USERS)
        protocol = "openai" if provider == "openai" else "anthropic"

        # 60% probability of cache hit for realistic proxy performance
        is_hit = random.random() < 0.62

        tokens_original = random.randint(150, 1800)
        if is_hit:
            cache_hit_count += 1
            tokens_sent = random.randint(20, 80)
            tokens_completion = random.randint(100, 600)
            tokens_saved = tokens_original - tokens_sent
            saved_usd = (tokens_saved / 1000.0) * in_cost
            cost_usd = (tokens_sent / 1000.0) * in_cost
            similarity = round(random.uniform(0.92, 0.999), 4)
            latency_ms = round(random.uniform(1.4, 4.8), 2)  # Sub-5ms cache hits
        else:
            tokens_sent = tokens_original
            tokens_completion = random.randint(100, 600)
            tokens_saved = 0
            saved_usd = 0.0
            cost_usd = (tokens_sent / 1000.0) * in_cost + (tokens_completion / 1000.0) * out_cost
            similarity = 0.0
            latency_ms = round(random.uniform(420.0, 1350.0), 2)  # Upstream latency

        # Some events triggered DLP redaction
        dlp_redactions = random.choice([0, 0, 0, 0, 1, 2, 3]) if random.random() < 0.25 else 0

        total_saved_usd += saved_usd
        total_cost_usd += cost_usd

        event_dict = {
            "id": str(uuid.uuid4()),
            "timestamp": event_time,
            "user_key": user_key,
            "provider": provider,
            "model": model,
            "protocol": protocol,
            "tokens_original": tokens_original,
            "tokens_sent": tokens_sent,
            "tokens_completion": tokens_completion,
            "tokens_saved": tokens_saved,
            "cost_usd": cost_usd,
            "saved_usd": saved_usd,
            "cache_hit": is_hit,
            "similarity": similarity,
            "dlp_redactions": dlp_redactions,
            "latency_ms": latency_ms,
        }
        await db_manager.record_analytics(event_dict)

    await db_manager.close()

    print("\n" + "-" * 60)
    print(" [OK] Seed completed successfully!")
    print(f" Total Events Seeded:     {total_events}")
    print(
        f" Cache Hits Generated:    {cache_hit_count} ({(cache_hit_count / total_events) * 100:.1f}%)"
    )
    print(f" Simulated Cost Saved:    ${total_saved_usd:.4f} USD")
    print(f" Simulated Upstream Cost: ${total_cost_usd:.4f} USD")
    print("-" * 60)
    print(" Open http://localhost:8787/dashboard to view the populated charts!\n")


def main() -> None:
    asyncio.run(seed())


if __name__ == "__main__":
    main()
