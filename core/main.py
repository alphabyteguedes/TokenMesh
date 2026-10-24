from __future__ import annotations

"""TokenMesh main application entry point.

Initializes the FastAPI application, lifespan context, CORS middleware,
dual-protocol AI Gateway routes (OpenAI and Anthropic), FinOps analytics,
and mounts the Linear.app-inspired dashboard static assets.
"""

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core.api.routes_anthropic import router as anthropic_router
from core.api.routes_finops import router as finops_router
from core.api.routes_openai import router as openai_router
from core.config import settings
from core.db.duckdb_manager import db_manager
from core.providers.anthropic_provider import anthropic_provider
from core.providers.openai_provider import openai_provider

logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tokenmesh")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle manager — handles DB init and connection teardown."""
    logger.info("TokenMesh AI Gateway starting on %s:%d ...", settings.HOST, settings.PORT)
    await db_manager.initialize()
    logger.info("DuckDB engine & VSS vector index successfully loaded.")
    yield
    logger.info("TokenMesh AI Gateway shutting down...")
    await db_manager.close()
    await openai_provider.close()
    await anthropic_provider.close()
    logger.info("All connections closed cleanly.")


app = FastAPI(
    title="TokenMesh",
    version="0.1.0",
    description="High-Performance Self-Hosted AI Gateway & Semantic Reverse Proxy",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routers (OpenAI native + cross-protocol, Anthropic native, FinOps analytics)
app.include_router(openai_router)
app.include_router(anthropic_router)
app.include_router(finops_router)


@app.get("/healthz", tags=["Health"])
async def health_check() -> JSONResponse:
    """Kubernetes / Docker health check probe."""
    return JSONResponse(
        content={
            "status": "healthy",
            "version": "0.1.0",
            "cache_enabled": settings.CACHE_ENABLED,
            "dlp_enabled": settings.DLP_ENABLED,
            "compression_enabled": settings.COMPRESSION_ENABLED,
        }
    )


@app.get("/dashboard", include_in_schema=False)
async def dashboard_page() -> FileResponse:
    """Serve the FinOps metrics dashboard."""
    dashboard_path = os.path.join(web_dir, "dashboard.html")
    return FileResponse(dashboard_path)


# Mount static web dashboard (MUST be mounted after all API routes)
web_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
if not os.path.isdir(web_dir):
    web_dir = "web"

if os.path.isdir(web_dir):
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "core.main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=False,
    )
