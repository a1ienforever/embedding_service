from typing import List

from schemas.base import Base


class EmbeddingRequest(Base):
    text: str

class EmbeddingBatchRequest(Base):
    texts: List[str]
