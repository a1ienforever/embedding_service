"""SentenceTransformer-based implementation of IEmbeddingService."""

from __future__ import annotations

import asyncio
import logging

from core.config import EmbeddingSettings
from core.errors import EmbeddingValidationError, EmbeddingServiceError

logger = logging.getLogger("smm_telegram.embedding")


class SentenceTransformerEmbeddingService:
    """Wraps ``sentence_transformers.SentenceTransformer`` as an async service."""

    def __init__(self, settings: EmbeddingSettings) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(settings.embedding_model_name)
            self._dim: int = self._model.get_sentence_embedding_dimension()
            logger.info(
                "Embedding model loaded: %s (dim=%d)",
                settings.embedding_model_name,
                self._dim,
            )
        except Exception as exc:
            raise EmbeddingServiceError(
                message=f"Failed to load embedding model '{settings.embedding_model_name}': {exc}",
                operation="load_model",
                cause=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Sync encode helpers (run inside a thread via asyncio.to_thread)
    # ------------------------------------------------------------------

    def _encode_single(self, text: str) -> list[float]:
        try:
            result = self._model.encode(text)
            return [float(v) for v in result]
        except Exception as exc:
            raise EmbeddingServiceError(
                message=f"Failed to encode text: {exc}",
                operation="encode",
                cause=exc,
            ) from exc

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            results = self._model.encode(texts)
            return [[float(v) for v in vec] for vec in results]
        except Exception as exc:
            raise EmbeddingServiceError(
                message=f"Failed to encode batch: {exc}",
                operation="encode_batch",
                cause=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Public async interface
    # ------------------------------------------------------------------

    async def embed(self, text: str) -> list[float]:
        if not text.strip():
            raise EmbeddingValidationError(
                message="Text must not be empty or whitespace-only.",
                text=text,
            )
        return await asyncio.to_thread(self._encode_single, text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        for text in texts:
            if not text.strip():
                raise EmbeddingValidationError(
                    message="All texts in batch must be non-empty.",
                    text=text,
                )
        return await asyncio.to_thread(self._encode_batch, texts)

    def dimension(self) -> int:
        return self._dim
