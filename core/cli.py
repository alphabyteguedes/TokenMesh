from __future__ import annotations

"""Command-line interface for the TokenMesh AI Gateway.

Provides operations to start the server, inspect live status, display FinOps
metrics in the terminal, and purge vector cache entries.
"""

import argparse
import asyncio
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

# Reconfigure standard output to UTF-8 on Windows consoles to prevent cp1252 encoding errors
reconfig = getattr(sys.stdout, "reconfigure", None)
if callable(reconfig):
    try:
        reconfig(encoding="utf-8")
        err_reconfig = getattr(sys.stderr, "reconfigure", None)
        if callable(err_reconfig):
            err_reconfig(encoding="utf-8")
    except Exception:
        pass

from core.config import settings


def _format_money(amount: float, symbol: str) -> str:
    """Format monetary value with two decimals and currency symbol."""
    return f"{symbol} {amount:,.2f}"


def _fetch_json(url: str, timeout: float = 3.0) -> dict[str, Any] | None:
    """Helper to fetch JSON via urllib."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "tokenmesh-cli/0.1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                parsed = json.loads(resp.read().decode("utf-8"))
                if isinstance(parsed, dict):
                    return parsed
    except Exception:
        return None
    return None


def cmd_start(args: argparse.Namespace) -> int:
    """Start the TokenMesh Gateway via uvicorn."""
    import uvicorn

    host: str = args.host or settings.HOST
    port: int = args.port or settings.PORT
    reload: bool = args.reload

    print("\n" + "=" * 60)
    print("  TOKENMESH AI GATEWAY & SEMANTIC REVERSE PROXY")
    print("  Zero-Cloud DuckDB VSS HNSW Cache | Local Zero-Trust DLP")
    print("=" * 60)
    print(f"  Target:     http://{host}:{port}")
    print(f"  Dashboard:  http://{host}:{port}/dashboard")
    print(f"  Playground: http://{host}:{port}/")
    print(f"  Database:   {settings.DUCKDB_PATH}")
    print("=" * 60 + "\n")

    uvicorn.run(
        "core.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Check health of the gateway and underlying database."""
    base_url: str = args.url.rstrip("/")
    health_url = f"{base_url}/healthz"

    print(f"Checking TokenMesh gateway status at {base_url}...")
    data = _fetch_json(health_url)

    if data and data.get("status") == "healthy":
        print("\n  [OK] STATUS: ONLINE (Healthy)")
        print(f"  Service:      {data.get('service', 'tokenmesh')}")
        print(f"  Version:      {data.get('version', '0.1.0')}")
        print(f"  DuckDB VSS:   {'Ready' if data.get('duckdb_vss') else 'Brute-force fallback'}")
        print(f"  Embedder:     {'Ready' if data.get('embedder_ready') else 'Pending'}")
        print(f"  Timestamp:    {data.get('timestamp')}")
        print(f"  Dashboard:    {base_url}/dashboard\n")
        return 0

    print("\n  [FAIL] STATUS: OFFLINE")
    print(f"  Could not reach gateway at {health_url}.")

    # Check local database file directly
    db_path = settings.DUCKDB_PATH
    if os.path.exists(db_path):
        size_kb = os.path.getsize(db_path) / 1024
        print(f"  Local DuckDB file exists: {db_path} ({size_kb:.1f} KB)")
    else:
        print(f"  Local DuckDB file not created yet: {db_path}")
    print("  Run 'tokenmesh start' to launch the gateway server.\n")
    return 1


def cmd_stats(args: argparse.Namespace) -> int:
    """Fetch and print live FinOps statistics in the terminal."""
    base_url: str = args.url.rstrip("/")
    currency: str = (args.currency or "USD").upper()
    url = f"{base_url}/v1/analytics/summary?currency={currency}"

    data = _fetch_json(url)
    if data is None:
        print(f"\n  \033[93m[!] Gateway not responding at {base_url}.\033[0m")
        print("  Reading directly from local DuckDB file...\n")
        from core.db.duckdb_manager import db_manager

        async def _local_summary() -> dict[str, Any]:
            await db_manager.initialize()
            res = await db_manager.get_summary(currency=currency)
            await db_manager.close()
            return dict(res)

        try:
            data = asyncio.run(_local_summary())
        except Exception as exc:
            print(f"  Error reading local database: {exc}")
            return 1

    symbol = data.get("currency_symbol", "$")
    total_req = data.get("total_requests", 0)
    cache_hits = data.get("cache_hits", 0)
    hit_rate = data.get("cache_hit_rate", 0.0)
    tokens_saved = data.get("total_tokens_saved", 0)
    total_saved = data.get("total_saved", 0.0)
    total_cost = data.get("total_cost", 0.0)
    dlp_redactions = data.get("total_dlp_redactions", 0)
    avg_latency = data.get("avg_latency_ms", 0.0)

    print("\n" + "+" + "-" * 58 + "+")
    print(f"|  TOKENMESH FINOPS & SEMANTIC LEDGER ({currency})".ljust(59) + "|")
    print("+" + "-" * 58 + "+")
    print(f"|  Total Requests:        {total_req:>12,d}".ljust(59) + "|")
    print(f"|  Semantic Cache Hits:   {cache_hits:>12,d} ({hit_rate:.1f}%)".ljust(59) + "|")
    print(f"|  Tokens Saved:          {tokens_saved:>12,d}".ljust(59) + "|")
    print(f"|  Financial Savings:     {_format_money(total_saved, symbol):>16}".ljust(59) + "|")
    print(f"|  Upstream API Spend:    {_format_money(total_cost, symbol):>16}".ljust(59) + "|")
    print(f"|  Local DLP Redactions:  {dlp_redactions:>12,d}".ljust(59) + "|")
    print(f"|  Average Latency:       {avg_latency:>13.2f} ms".ljust(59) + "|")
    print("+" + "-" * 58 + "+\n")
    return 0


def cmd_purge_cache(args: argparse.Namespace) -> int:
    """Purge entries from semantic cache."""
    older_than: int | None = args.older_than
    confirm: bool = args.yes

    if not confirm:
        scope = f"older than {older_than} days" if older_than else "ALL entries"
        ans = input(f"Are you sure you want to purge {scope} from semantic cache? [y/N]: ")
        if ans.lower() not in ("y", "yes"):
            print("Purge aborted.")
            return 0

    from core.db.duckdb_manager import db_manager

    async def _purge() -> int:
        await db_manager.initialize()
        count = await db_manager.purge_cache(older_than_days=older_than)
        await db_manager.close()
        return count

    count = asyncio.run(_purge())
    print(f"\n  [OK] Purged {count} cached vector entries from DuckDB.\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(
        prog="tokenmesh",
        description="TokenMesh AI Gateway & Semantic Reverse Proxy CLI",
    )
    parser.add_argument("--version", action="version", version="TokenMesh 0.1.0 (Apache 2.0)")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Start command
    p_start = subparsers.add_parser("start", help="Start the gateway server")
    p_start.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)"
    )
    p_start.add_argument("--port", type=int, default=8787, help="Port to listen on (default: 8787)")
    p_start.add_argument("--reload", action="store_true", help="Enable auto-reload on file change")

    # Status command
    p_status = subparsers.add_parser("status", help="Check gateway health and state")
    p_status.add_argument(
        "--url", type=str, default="http://localhost:8787", help="Base URL of gateway"
    )

    # Stats command
    p_stats = subparsers.add_parser("stats", help="Show live FinOps metrics in the terminal")
    p_stats.add_argument(
        "--url", type=str, default="http://localhost:8787", help="Base URL of gateway"
    )
    p_stats.add_argument(
        "--currency",
        type=str,
        default="USD",
        help="Currency code (USD, BRL, EUR, GBP, JPY, CAD, CHF, AUD, MXN, ARS)",
    )

    # Purge cache
    p_purge = subparsers.add_parser("purge-cache", help="Purge vector entries from semantic cache")
    p_purge.add_argument(
        "--older-than", type=int, default=None, help="Purge entries older than N days"
    )
    p_purge.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the tokenmesh CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "start":
        return cmd_start(args)
    if args.command == "status":
        return cmd_status(args)
    if args.command == "stats":
        return cmd_stats(args)
    if args.command == "purge-cache":
        return cmd_purge_cache(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
