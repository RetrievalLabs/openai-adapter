"""
Copyright (c) 2026 RetrievalLabs Co. All rights reserved.
Licensed under the Apache License, Version 2.0.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from rag_control.exceptions import QueryEmbeddingAdapterError
from rag_control.models import QueryEmbeddingResponse
from rag_control.models.user_context import UserContext

from openai_adapter.query_embedding.adapter import OpenAIQueryEmbeddingAdapter


class TestOpenAIQueryEmbeddingAdapter:
    """Test suite for OpenAIQueryEmbeddingAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create an adapter instance with mocked OpenAI client."""
        with patch("openai_adapter.query_embedding.adapter.OpenAI") as mock:
            mock.return_value = MagicMock()
            adapter = OpenAIQueryEmbeddingAdapter(api_key="test-key", model="text-embedding-3-small")
            yield adapter

    def test_initialization_success(self):
        """Test successful adapter initialization."""
        with patch("openai_adapter.query_embedding.adapter.OpenAI") as mock:
            mock.return_value = MagicMock()
            adapter = OpenAIQueryEmbeddingAdapter(api_key="test-key")
            assert adapter.embedding_model == "text-embedding-3-small"
            mock.assert_called_once_with(api_key="test-key")

    def test_initialization_with_custom_model(self):
        """Test initialization with custom embedding model."""
        with patch("openai_adapter.query_embedding.adapter.OpenAI") as mock:
            mock.return_value = MagicMock()
            adapter = OpenAIQueryEmbeddingAdapter(
                api_key="test-key", model="text-embedding-3-large"
            )
            assert adapter.embedding_model == "text-embedding-3-large"
            mock.assert_called_once_with(api_key="test-key")

    def test_initialization_with_kwargs(self):
        """Test initialization with additional OpenAI client options."""
        with patch("openai_adapter.query_embedding.adapter.OpenAI") as mock:
            mock.return_value = MagicMock()
            adapter = OpenAIQueryEmbeddingAdapter(
                api_key="test-key",
                model="text-embedding-3-small",
                organization="test-org",
                timeout=30,
            )
            assert adapter.embedding_model == "text-embedding-3-small"
            mock.assert_called_once_with(
                api_key="test-key", organization="test-org", timeout=30
            )

    def test_initialization_failure(self):
        """Test initialization failure handling."""
        with patch("openai_adapter.query_embedding.adapter.OpenAI") as mock:
            mock.side_effect = Exception("Connection error")
            with pytest.raises(QueryEmbeddingAdapterError, match="Failed to initialize OpenAI client"):
                OpenAIQueryEmbeddingAdapter(api_key="test-key")

    def test_embedding_model_property(self, adapter):
        """Test embedding_model property."""
        assert adapter.embedding_model == "text-embedding-3-small"

    def _create_mock_embedding_response(self, embedding_dim=1536):
        """Helper to create mock embedding response."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1] * embedding_dim
        mock_response.usage = MagicMock(prompt_tokens=10)
        mock_response.model = "text-embedding-3-small"
        return mock_response

    def test_embed_basic(self, adapter):
        """Test basic embed functionality."""
        mock_response = self._create_mock_embedding_response()
        adapter._client.embeddings.create.return_value = mock_response

        result = adapter.embed("What is AI?")

        assert isinstance(result, QueryEmbeddingResponse)
        assert len(result.embedding) == 1536
        assert result.metadata.model == "text-embedding-3-small"
        assert result.metadata.provider == "openai"
        assert result.metadata.dimensions == 1536
        assert isinstance(result.metadata.timestamp, datetime)
        assert result.metadata.latency_ms > 0

    @pytest.mark.parametrize(
        "query,embedding_dim",
        [
            ("What is machine learning?", 1536),
            ("Short query", 1536),
            ("Very long query " * 100, 1536),
            ("Emoji test 🚀", 1536),
            ("Multi-language 中文", 1536),
        ],
    )
    def test_embed_various_queries(self, adapter, query, embedding_dim):
        """Test embed with various query types."""
        mock_response = self._create_mock_embedding_response(embedding_dim)
        adapter._client.embeddings.create.return_value = mock_response

        result = adapter.embed(query)

        assert result.embedding is not None
        assert len(result.embedding) == embedding_dim
        assert result.metadata.dimensions == embedding_dim
        adapter._client.embeddings.create.assert_called_once()
        call_kwargs = adapter._client.embeddings.create.call_args[1]
        assert call_kwargs["input"] == query

    def test_embed_response_structure(self, adapter):
        """Test embed response contains all required fields."""
        mock_response = self._create_mock_embedding_response()
        adapter._client.embeddings.create.return_value = mock_response

        result = adapter.embed("Test")

        assert isinstance(result.embedding, list)
        assert len(result.embedding) == 1536
        assert result.metadata.model == "text-embedding-3-small"
        assert result.metadata.provider == "openai"
        assert result.metadata.dimensions == 1536
        assert result.metadata.latency_ms > 0
        assert "model" in result.metadata.raw
        assert "usage" in result.metadata.raw

    def test_embed_different_dimensions(self, adapter):
        """Test embed with different embedding dimensions."""
        for dim in [256, 512, 1536, 3072]:
            mock_response = self._create_mock_embedding_response(embedding_dim=dim)
            adapter._client.embeddings.create.return_value = mock_response

            result = adapter.embed("Test")

            assert len(result.embedding) == dim
            assert result.metadata.dimensions == dim

    def test_embed_api_error(self, adapter):
        """Test embed error handling."""
        adapter._client.embeddings.create.side_effect = Exception("API error")

        with pytest.raises(QueryEmbeddingAdapterError, match="Failed to get embedding from OpenAI"):
            adapter.embed("Test")

    def test_embed_user_context(self, adapter):
        """Test embed accepts user_context parameter."""
        mock_response = self._create_mock_embedding_response()
        adapter._client.embeddings.create.return_value = mock_response

        user_context = UserContext(user_id="test_user", org_id="test_org", attributes={})
        result = adapter.embed("Test", user_context=user_context)

        assert result.embedding is not None

    def test_embed_raw_metadata(self, adapter):
        """Test that embed includes raw response metadata."""
        mock_response = self._create_mock_embedding_response()
        adapter._client.embeddings.create.return_value = mock_response

        result = adapter.embed("Test")

        assert result.metadata.raw["model"] == "text-embedding-3-small"
        assert result.metadata.raw["usage"] is not None

    def test_embed_api_call_parameters(self, adapter):
        """Test that embed passes correct parameters to API."""
        mock_response = self._create_mock_embedding_response()
        adapter._client.embeddings.create.return_value = mock_response

        adapter.embed("Test query")

        adapter._client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input="Test query"
        )

    def test_embed_multiple_calls(self, adapter):
        """Test multiple embed calls."""
        mock_response = self._create_mock_embedding_response()
        adapter._client.embeddings.create.return_value = mock_response

        queries = ["Query 1", "Query 2", "Query 3"]
        results = [adapter.embed(q) for q in queries]

        assert len(results) == 3
        for result in results:
            assert len(result.embedding) == 1536
            assert result.metadata.dimensions == 1536

        assert adapter._client.embeddings.create.call_count == 3

    def test_embed_timestamp_uniqueness(self, adapter):
        """Test that each embed call has its own timestamp."""
        import time

        mock_response = self._create_mock_embedding_response()
        adapter._client.embeddings.create.return_value = mock_response

        result1 = adapter.embed("Query 1")
        time.sleep(0.01)
        result2 = adapter.embed("Query 2")

        assert result1.metadata.timestamp <= result2.metadata.timestamp

    def test_embed_latency_captured(self, adapter):
        """Test that embed captures latency measurement."""
        mock_response = self._create_mock_embedding_response()
        adapter._client.embeddings.create.return_value = mock_response

        result = adapter.embed("Test")

        assert result.metadata.latency_ms > 0
        assert isinstance(result.metadata.latency_ms, float)

    def test_embed_with_large_embedding(self, adapter):
        """Test embed with large embedding vector."""
        mock_response = self._create_mock_embedding_response(embedding_dim=3072)
        adapter._client.embeddings.create.return_value = mock_response

        result = adapter.embed("Test")

        assert len(result.embedding) == 3072
        assert result.metadata.dimensions == 3072

    def test_embed_consistency(self, adapter):
        """Test that same query produces same structure (not same values)."""
        mock_response1 = self._create_mock_embedding_response()
        mock_response1.data[0].embedding = [0.1, 0.2, 0.3] + [0.0] * 1533

        mock_response2 = self._create_mock_embedding_response()
        mock_response2.data[0].embedding = [0.1, 0.2, 0.3] + [0.0] * 1533

        adapter._client.embeddings.create.side_effect = [mock_response1, mock_response2]

        result1 = adapter.embed("Test")
        result2 = adapter.embed("Test")

        assert len(result1.embedding) == len(result2.embedding)
        assert result1.metadata.model == result2.metadata.model
        assert result1.metadata.provider == result2.metadata.provider

    def test_embed_none_usage(self, adapter):
        """Test embed when response has no usage information."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1] * 1536
        mock_response.usage = None
        mock_response.model = "text-embedding-3-small"

        adapter._client.embeddings.create.return_value = mock_response

        result = adapter.embed("Test")

        assert result.embedding is not None
        assert result.metadata.raw["usage"] is None
