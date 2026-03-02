import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from core.config import EmbeddingSettings
from core.errors import EmbeddingServiceError
from emdedding.service import SentenceTransformerEmbeddingService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        embedding_settings = EmbeddingSettings.model_validate({})
        app.state.embedding_service = SentenceTransformerEmbeddingService(settings=embedding_settings)
    except EmbeddingServiceError as e:
        logger.error("Failed to initialize embedding service: %s", e)
        app.state.embedding_service = None
    yield

app = FastAPI(
    title="Embedding Service",
    description="API Client for work with embedding model",
    version="0.1.0",
    lifespan=lifespan,
)

from api.get import router as get_router
from api.post import router as post_router

app.include_router(get_router)
app.include_router(post_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)