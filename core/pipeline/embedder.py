"""Local CPU ONNX Embedder using fastembed.

Provides a lightweight, async-safe embedding interface that runs entirely on
the local CPU via ONNX Runtime.  Model loading is lazy and thread-safe so the
first ``embed`` call incurs the one-time download/load cost.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedder class
# ---------------------------------------------------------------------------


class LocalEmbedder:
    """Async wrapper around ``fastembed.TextEmbedding`` for CPU inference."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._model: TextEmbedding | None = None
        self._init_lock: asyncio.Lock = asyncio.Lock()
        self._initialized: bool = False

    # -- private helpers ----------------------------------------------------

    def _load_model(self) -> None:
        """Synchronously download (if needed) and load the embedding model."""
        try:
            from fastembed import TextEmbedding  # noqa: WPS433 — lazy import

            logger.info("Loading fastembed model '%s' …", self._model_name)
            self._model = TextEmbedding(model_name=self._model_name)
            logger.info("Model '%s' loaded successfully.", self._model_name)
        except Exception as exc:
            logger.warning(
                "Could not initialize fastembed model (%s). Using deterministic CPU fallback.",
                exc,
            )
            self._model = None

    async def _ensure_model(self) -> None:
        """Lazily initialise the model, guarded by an async lock."""
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await asyncio.to_thread(self._load_model)
            self._initialized = True

    def _deterministic_embed(self, text: str) -> list[float]:
        """Deterministic normalized 384-dimensional fallback vector."""
        import hashlib
        import math

        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
        vec: list[float] = []
        for _ in range(384):
            seed = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            val = ((seed % 100000) / 100000.0) - 0.5
            vec.append(val)
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [round(x / norm, 6) for x in vec]

    def _sync_embed(self, text: str) -> list[float]:
        """Run a single embedding synchronously (called via ``to_thread``)."""
        if self._model is not None:
            try:
                embeddings = list(self._model.embed([text]))
                return [float(x) for x in embeddings[0].tolist()]
            except Exception as e:
                logger.warning("fastembed embed failed (%s), falling back", e)
        return self._deterministic_embed(text)

    def _sync_embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Run a batch embedding synchronously (called via ``to_thread``)."""
        if self._model is not None:
            try:
                embeddings = list(self._model.embed(texts))
                return [[float(x) for x in e.tolist()] for e in embeddings]
            except Exception as e:
                logger.warning("fastembed embed_batch failed (%s), falling back", e)
        return [self._deterministic_embed(t) for t in texts]

    # -- public API ---------------------------------------------------------

    async def embed(self, text: str) -> list[float]:
        """Embed a single text and return a vector of 384 floats."""
        await self._ensure_model()
        return await asyncio.to_thread(self._sync_embed, text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts and return a list of 384-float vectors."""
        await self._ensure_model()
        return await asyncio.to_thread(self._sync_embed_batch, texts)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
embedder = LocalEmbedder()
