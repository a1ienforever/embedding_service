from fastapi import APIRouter, Depends, HTTPException

from emdedding.di import get_embedding_service
from emdedding.service import SentenceTransformerEmbeddingService
from schemas.response import EmbeddingDimResponse

router = APIRouter(prefix="/config", tags=["config"])

@router.get("/dim")
async def get_dim(
        embedding_service: SentenceTransformerEmbeddingService = Depends(get_embedding_service)
) -> EmbeddingDimResponse:
    if embedding_service is None:
        raise HTTPException(status_code=503, detail="Embedding service is not available")
    return EmbeddingDimResponse(dim=embedding_service.dimension())