import time
from datetime import datetime
from typing import Any

from openai import OpenAI
from rag_control.adapters import QueryEmbedding
from rag_control.exceptions import QueryEmbeddingAdapterError
from rag_control.models import (
    QueryEmbeddingMetadata,
    QueryEmbeddingResponse,
)


class OpenAIQueryEmbeddingAdapter(QueryEmbedding):
    """
    OpenAI adapter for query embedding.

    Implements the QueryEmbedding interface to generate embeddings for text queries
    using OpenAI's embedding models. Supports full OpenAI client configuration options.

    Example:
        adapter = OpenAIQueryEmbeddingAdapter(api_key="sk-...")
        response = adapter.embed("What is machine learning?")
    """

    def __init__(self, api_key: str, model: str = "text-embedding-3-small", **kwargs: Any) -> None:
        """
        Initialize the OpenAI embedding adapter.

        Args:
            api_key: OpenAI API key for authentication
            model: Embedding model to use (default: text-embedding-3-small)
            **kwargs: Additional OpenAI client configuration options
                (e.g., organization, project, base_url, timeout)

        Raises:
            QueryEmbeddingAdapterError: If OpenAI client initialization fails
        """
        self._model = model
        try:
            self._client = OpenAI(api_key=api_key, **kwargs)
        except Exception as e:
            raise QueryEmbeddingAdapterError(f"Failed to initialize OpenAI client: {str(e)}")

    @property
    def embedding_model(self) -> str:
        """Return the name of the embedding model being used."""
        return self._model

    def embed(self, query: str) -> QueryEmbeddingResponse:
        """
        Generate an embedding vector for the given query.

        Calls the OpenAI embeddings API and returns the embedding along with
        comprehensive metadata including latency, dimensions, and request details.

        Args:
            query: The text to embed

        Returns:
            QueryEmbeddingResponse: Contains the embedding vector and metadata
                (model, provider, latency, dimensions, request_id, timestamp)

        Raises:
            QueryEmbeddingAdapterError: If the embedding API call fails
        """
        try:
            # Measure API latency
            start_time = time.time()
            response = self._client.embeddings.create(model=self._model, input=query)
            latency_ms = (time.time() - start_time) * 1000

            # Extract embedding vector and dimensions
            embedding = response.data[0].embedding
            dimensions = len(embedding)

            # Collect comprehensive metadata about the embedding request
            metadata = QueryEmbeddingMetadata(
                model=self._model,
                provider="openai",
                latency_ms=latency_ms,
                dimensions=dimensions,
                timestamp=datetime.utcnow(),
                raw={
                    "model": response.model,
                    "usage": response.usage.model_dump() if response.usage else None,
                },
            )

            return QueryEmbeddingResponse(embedding=embedding, metadata=metadata)
        except Exception as e:
            raise QueryEmbeddingAdapterError(f"Failed to get embedding from OpenAI: {str(e)}")
