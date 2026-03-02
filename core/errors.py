"""Domain exceptions for the embedding service."""

from __future__ import annotations


class EmbeddingError(Exception):
    """Base exception for all embedding service errors."""


class EmbeddingValidationError(EmbeddingError):
    """Raised when input text is empty or consists only of whitespace."""

    def __init__(self, message: str, text: str) -> None:
        super().__init__(message)
        self.text = text


class EmbeddingServiceError(EmbeddingError):
    """Raised on model load failure or encode error."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.cause = cause
