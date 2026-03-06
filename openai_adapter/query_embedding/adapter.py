
import time
from datetime import datetime
from openai import OpenAI
from rag_control.adapters import QueryEmbedding
from rag_control.models import (
    QueryEmbeddingMetadata,
    QueryEmbeddingResponse,
)
from rag_control.exceptions import QueryEmbeddingAdapterError

class OpenAIQueryEmbeddingAdapter(QueryEmbedding):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", **kwargs):
        self._model = model
        try:
            self._client = OpenAI(api_key=api_key, **kwargs)
        except Exception as e:
            raise QueryEmbeddingAdapterError(f"Failed to initialize OpenAI client: {str(e)}")
        
    @property
    def embedding_model(self) -> str:
        return self._model

    def embed(self, query: str) -> QueryEmbeddingResponse:
        try:
            start_time = time.time()
            response = self._client.embeddings.create(
                model=self._model,
                input=query
            )
            latency_ms = (time.time() - start_time) * 1000

            embedding = response.data[0].embedding
            dimensions = len(embedding)

            metadata = QueryEmbeddingMetadata(
                model=self._model,
                provider="openai",
                latency_ms=latency_ms,
                dimensions=dimensions,
                request_id=response.id,
                timestamp=datetime.utcnow(),
                raw={
                    "model": response.model,
                    "usage": response.usage.model_dump() if response.usage else None,
                }
            )

            return QueryEmbeddingResponse(embedding=embedding, metadata=metadata)
        except Exception as e:
            raise QueryEmbeddingAdapterError(f"Failed to get embedding from OpenAI: {str(e)}")