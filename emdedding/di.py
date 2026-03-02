from fastapi import Request

from emdedding.service import SentenceTransformerEmbeddingService

def get_embedding_service(request: Request) -> SentenceTransformerEmbeddingService | None:
    """Get SentenceTransformerEmbeddingService from app state.

    Returns None if not configured (graceful degradation).

    Args:
        request: FastAPI request object

    Returns:
        SentenceTransformerEmbeddingService instance or None
    """
    return getattr(request.app.state, "embedding_service", None)
