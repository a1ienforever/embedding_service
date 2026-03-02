from schemas.base import Base


class EmbeddingResponse(Base):
    vector: list[float]

class EmbeddingBatchResponse(Base):
    vectors: list[list[float]]

class EmbeddingDimResponse(Base):
    dim: int
