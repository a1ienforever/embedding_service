from fastapi.routing import APIRouter
from fastapi import Depends, HTTPException

from emdedding.di import get_embedding_service
from emdedding.service import SentenceTransformerEmbeddingService
from schemas.request import EmbeddingRequest, EmbeddingBatchRequest
from schemas.response import EmbeddingResponse, EmbeddingBatchResponse

from core.errors import EmbeddingValidationError, EmbeddingServiceError

router = APIRouter(prefix="/embed", tags=["embed"])


@router.post("", response_model=EmbeddingResponse)
async def embed(request: EmbeddingRequest,
                embedding_service: SentenceTransformerEmbeddingService = Depends(get_embedding_service)
                ) -> EmbeddingResponse:
    try:
        vector: list[float] = await embedding_service.embed(text=request.text)
        return EmbeddingResponse(vector=vector)
    except EmbeddingValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except EmbeddingServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=EmbeddingBatchResponse)
async def embed_batch(request: EmbeddingBatchRequest,
                      embedding_service: SentenceTransformerEmbeddingService = Depends(get_embedding_service)
                      ) -> EmbeddingBatchResponse:
    try:
        vectors: list[list[float]] = await embedding_service.embed_batch(texts=request.texts)
        return EmbeddingBatchResponse(vectors=vectors)
    except EmbeddingValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except EmbeddingServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))