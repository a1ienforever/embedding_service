"""Infrastructure configuration for the embedding service."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

class EmbeddingSettings(BaseSettings):
    """Embedding model configuration loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    embedding_model_name: str = "ai-forever/ru-en-RoSBERTa"
