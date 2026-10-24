from __future__ import annotations

import asyncio
import os
import tempfile
from collections.abc import AsyncGenerator, Generator
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Set test environment variables BEFORE importing app modules.
# These ensure the application bootstraps with safe, deterministic settings
# regardless of what the developer has in their local `.env`.
# ---------------------------------------------------------------------------
os.environ["TOKENMESH_DUCKDB_PATH"] = ""  # Will be overridden per test
os.environ["TOKENMESH_OPENAI_API_KEY"] = "sk-test-key-for-unit-tests-only"
os.environ["TOKENMESH_ANTHROPIC_API_KEY"] = "sk-ant-test-key-for-unit-tests-only"
os.environ["TOKENMESH_CACHE_ENABLED"] = "true"
os.environ["TOKENMESH_DLP_ENABLED"] = "true"

from core.db.duckdb_manager import DuckDBManager  # noqa: E402
from core.main import app  # noqa: E402


# ---------------------------------------------------------------------------
# Session-scoped event loop (required for session/module fixtures with
# pytest-asyncio ≥ 0.21).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create a new event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Per-test temporary DuckDB instance.
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture
async def tmp_db() -> AsyncGenerator[DuckDBManager, None]:
    """Create a temporary DuckDB database for test isolation.

    A new temp file is created for each test, guaranteeing a clean
    database state.  The manager is fully initialised (tables & VSS
    extension loaded) before being yielded to the test.
    """
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

    # Teardown: close connection, remove temp file.
    await db.close()
    try:
        os.unlink(tmp_path)
    except OSError:
        pass
    # DuckDB may create a WAL sidecar file; clean it up too.
    wal_path = f"{tmp_path}.wal"
    try:
        os.unlink(wal_path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# HTTPX async test client wired to the FastAPI ASGI app.
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture
async def client(tmp_db: DuckDBManager) -> AsyncGenerator[AsyncClient, None]:
    """Provide an ``AsyncClient`` bound to the TokenMesh ASGI app.

    The global ``db_manager`` singleton is monkey-patched with the
    per-test ``tmp_db`` instance so that every test operates on its
    own isolated database.
    """
    with (
        patch("core.api.routes_openai.db_manager", tmp_db),
        patch("core.api.routes_anthropic.db_manager", tmp_db),
        patch("core.api.routes_finops.db_manager", tmp_db),
        patch("core.db.duckdb_manager.db_manager", tmp_db),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            yield ac
