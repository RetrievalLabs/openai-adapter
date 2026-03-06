"""
Copyright (c) 2026 RetrievalLabs Co. All rights reserved.
Licensed under the Apache License, Version 2.0.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from rag_control.exceptions import LLMAdapterError
from rag_control.models import LLMResponse
from rag_control.models.user_context import UserContext

from openai_adapter.llm.adapter import OpenAILLMAdapter


class TestOpenAILLMAdapter:
    """Test suite for OpenAILLMAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create an adapter instance with mocked OpenAI client."""
        with patch("openai_adapter.llm.adapter.OpenAI") as mock:
            mock.return_value = MagicMock()
            adapter = OpenAILLMAdapter(api_key="test-key", model="gpt-4")
            yield adapter

    def test_initialization_success(self):
        """Test successful adapter initialization."""
        with patch("openai_adapter.llm.adapter.OpenAI") as mock:
            mock.return_value = MagicMock()
            adapter = OpenAILLMAdapter(api_key="test-key", model="gpt-4")
            assert adapter.model_name == "gpt-4"
            mock.assert_called_once_with(api_key="test-key")

    def test_initialization_with_kwargs(self):
        """Test initialization with additional OpenAI client options."""
        with patch("openai_adapter.llm.adapter.OpenAI") as mock:
            mock.return_value = MagicMock()
            adapter = OpenAILLMAdapter(
                api_key="test-key",
                model="gpt-4",
                organization="test-org",
                timeout=30,
            )
            assert adapter.model_name == "gpt-4"
            mock.assert_called_once_with(
                api_key="test-key", organization="test-org", timeout=30
            )

    def test_initialization_failure(self):
        """Test initialization failure handling."""
        with patch("openai_adapter.llm.adapter.OpenAI") as mock:
            mock.side_effect = Exception("Connection error")
            with pytest.raises(LLMAdapterError, match="Failed to initialize OpenAI client"):
                OpenAILLMAdapter(api_key="test-key")

    def test_model_name_property(self, adapter):
        """Test model_name property."""
        assert adapter.model_name == "gpt-4"

    def _create_mock_response(self, content="Generated text", finish_reason="stop"):
        """Helper to create mock response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = content
        mock_response.choices[0].finish_reason = finish_reason
        mock_response.usage = MagicMock(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
        mock_response.id = "chatcmpl-123"
        mock_response.model = "gpt-4"
        return mock_response

    @pytest.mark.parametrize(
        "prompt,expected_messages",
        [
            ("What is AI?", [{"role": "user", "content": "What is AI?"}]),
            (
                [{"role": "system", "content": "Help"}, {"role": "user", "content": "Test"}],
                [{"role": "system", "content": "Help"}, {"role": "user", "content": "Test"}],
            ),
        ],
    )
    def test_generate_prompt_types(self, adapter, prompt, expected_messages):
        """Test generate with different prompt types."""
        adapter._client.chat.completions.create.return_value = self._create_mock_response()

        result = adapter.generate(prompt)

        assert isinstance(result, LLMResponse)
        assert result.content == "Generated text"
        call_kwargs = adapter._client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"] == expected_messages

    @pytest.mark.parametrize("temperature,max_tokens", [(0.5, None), (None, 100), (0.8, 50)])
    def test_generate_parameters(self, adapter, temperature, max_tokens):
        """Test generate with various parameter combinations."""
        adapter._client.chat.completions.create.return_value = self._create_mock_response()

        result = adapter.generate("Test", temperature=temperature, max_output_tokens=max_tokens)

        assert result.content == "Generated text"
        call_kwargs = adapter._client.chat.completions.create.call_args[1]
        if temperature is not None:
            assert call_kwargs["temperature"] == temperature
        if max_tokens is not None:
            assert call_kwargs["max_tokens"] == max_tokens

    def test_generate_response_structure(self, adapter):
        """Test generate response contains all required fields."""
        adapter._client.chat.completions.create.return_value = self._create_mock_response()

        result = adapter.generate("Test")

        assert result.content == "Generated text"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20
        assert result.usage.total_tokens == 30
        assert result.metadata.model == "gpt-4"
        assert result.metadata.provider == "openai"
        assert result.metadata.request_id == "chatcmpl-123"
        assert isinstance(result.metadata.timestamp, datetime)
        assert result.metadata.latency_ms > 0
        assert result.metadata.raw["finish_reason"] == "stop"

    def test_generate_without_usage(self, adapter):
        """Test generate when response has no usage information."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = None
        mock_response.id = "chatcmpl-111"
        mock_response.model = "gpt-4"

        adapter._client.chat.completions.create.return_value = mock_response

        result = adapter.generate("Test")

        assert result.content == "Response"
        assert result.usage.prompt_tokens == 0
        assert result.usage.completion_tokens == 0
        assert result.usage.total_tokens == 0

    def test_generate_api_error(self, adapter):
        """Test generate error handling for API failures."""
        adapter._client.chat.completions.create.side_effect = Exception("API error")

        with pytest.raises(LLMAdapterError, match="Failed to generate text from OpenAI"):
            adapter.generate("Test")

    def test_generate_user_context(self, adapter):
        """Test generate accepts user_context parameter."""
        adapter._client.chat.completions.create.return_value = self._create_mock_response()

        user_context = UserContext(user_id="test_user", org_id="test_org", attributes={})
        result = adapter.generate("Test", user_context=user_context)

        assert result.content == "Generated text"

    def _create_mock_stream_chunks(self, include_usage=True):
        """Helper to create mock stream chunks."""
        chunks = []

        chunk1 = MagicMock()
        chunk1.id = "chatcmpl-stream1"
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello "
        chunk1.usage = None
        chunks.append(chunk1)

        chunk2 = MagicMock()
        chunk2.id = "chatcmpl-stream1"
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = "world"
        chunk2.usage = None
        chunks.append(chunk2)

        chunk3 = MagicMock()
        chunk3.id = "chatcmpl-stream1"
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].delta.content = None
        chunk3.usage = (
            MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
            if include_usage
            else None
        )
        chunks.append(chunk3)

        return chunks

    @pytest.mark.parametrize(
        "prompt",
        [
            "Hi",
            [{"role": "user", "content": "Hi"}],
        ],
    )
    def test_stream_prompt_types(self, adapter, prompt):
        """Test stream with different prompt types."""
        chunks = self._create_mock_stream_chunks(include_usage=False)
        adapter._client.chat.completions.create.return_value = iter(chunks)

        result = adapter.stream(prompt)
        stream_chunks = list(result.stream)

        assert result.metadata.provider == "openai"
        assert len(stream_chunks) == 2
        assert stream_chunks[0].delta == "Hello "
        assert stream_chunks[1].delta == "world"

    @pytest.mark.parametrize("temperature,max_tokens", [(0.5, None), (None, 100), (0.8, 50)])
    def test_stream_parameters(self, adapter, temperature, max_tokens):
        """Test stream with various parameter combinations."""
        chunks = self._create_mock_stream_chunks()
        adapter._client.chat.completions.create.return_value = iter(chunks)

        result = adapter.stream("Test", temperature=temperature, max_output_tokens=max_tokens)
        list(result.stream)

        call_kwargs = adapter._client.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True
        if temperature is not None:
            assert call_kwargs["temperature"] == temperature
        if max_tokens is not None:
            assert call_kwargs["max_tokens"] == max_tokens

    def test_stream_empty_chunks_skipped(self, adapter):
        """Test that empty/None chunks are skipped in stream."""
        chunk1 = MagicMock()
        chunk1.id = "chatcmpl-empty"
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Text"
        chunk1.usage = None

        chunk2 = MagicMock()
        chunk2.id = "chatcmpl-empty"
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = None
        chunk2.usage = None

        chunk3 = MagicMock()
        chunk3.id = "chatcmpl-empty"
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].delta.content = ""
        chunk3.usage = None

        adapter._client.chat.completions.create.return_value = iter([chunk1, chunk2, chunk3])

        result = adapter.stream("Test")
        chunks = list(result.stream)

        assert len(chunks) == 1
        assert chunks[0].delta == "Text"

    def test_stream_api_error(self, adapter):
        """Test stream error handling."""
        adapter._client.chat.completions.create.side_effect = Exception("Stream error")

        with pytest.raises(LLMAdapterError, match="Failed to stream from OpenAI"):
            adapter.stream("Test")

    def test_stream_user_context(self, adapter):
        """Test stream accepts user_context parameter."""
        chunks = self._create_mock_stream_chunks()
        adapter._client.chat.completions.create.return_value = iter(chunks)

        user_context = UserContext(user_id="test_user", org_id="test_org", attributes={})
        result = adapter.stream("Test", user_context=user_context)
        list(result.stream)

        assert result.metadata.model == "gpt-4"
        assert result.metadata.provider == "openai"

    def test_stream_metadata(self, adapter):
        """Test stream metadata structure."""
        chunks = self._create_mock_stream_chunks()
        adapter._client.chat.completions.create.return_value = iter(chunks)

        result = adapter.stream("Test")
        list(result.stream)

        assert result.metadata.model == "gpt-4"
        assert result.metadata.provider == "openai"
        assert isinstance(result.metadata.timestamp, datetime)
        assert result.metadata.latency_ms > 0
