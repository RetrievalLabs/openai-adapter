"""
Copyright (c) 2026 RetrievalLabs Co. All rights reserved.
Licensed under the Apache License, Version 2.0.
"""

import time
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI
from rag_control.adapters import LLM
from rag_control.exceptions import LLMAdapterError
from rag_control.models import (
    LLMMetadata,
    LLMResponse,
    LLMStreamChunk,
    LLMStreamResponse,
    LLMUsage,
    PromptInput,
)
from rag_control.models.user_context import UserContext


class OpenAILLMAdapter(LLM):  
    """
    OpenAI adapter for language models.

    Implements the LLM interface to generate text completions using OpenAI's language models.
    Supports full OpenAI client configuration options and provides detailed metadata about
    the model, usage, and response.

    Example:
        adapter = OpenAILLMAdapter(api_key="sk-...")
        response = adapter.generate("What is machine learning?")
    """

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", **kwargs: Any) -> None:
        """
        Initialize the OpenAI LLM adapter.

        Args:
            api_key: OpenAI API key for authentication
            model: Language model to use (default: gpt-3.5-turbo)
            **kwargs: Additional OpenAI client configuration options
                (e.g., organization, project, base_url, timeout)

        Raises:
            LLMAdapterError: If OpenAI client initialization fails
        """
        self._model = model
        try:
            self._client = OpenAI(api_key=api_key, **kwargs)
        except Exception as e:
            raise LLMAdapterError(f"Failed to initialize OpenAI client: {str(e)}")

    @property
    def model_name(self) -> str:
        """Return the name of the language model being used."""
        return self._model

    def generate(
        self,
        prompt: PromptInput,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        user_context: UserContext | None = None,
    ) -> LLMResponse:
        """
        Generate text completion for the given prompt.

        Args:
            prompt: String or list of ChatMessage objects for text generation
            temperature: Sampling temperature (0-2)
            max_output_tokens: Maximum tokens in response
            user_context: User context for the request

        Returns:
            LLMResponse: Contains generated text and metadata

        Raises:
            LLMAdapterError: If the API call fails
        """
        try:
            start_time = time.time()
            # Convert string prompt to messages format if needed
            messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt

            # Build API parameters
            api_params: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
            }
            if temperature is not None:
                api_params["temperature"] = temperature
            if max_output_tokens is not None:
                api_params["max_tokens"] = max_output_tokens

            response = self._client.chat.completions.create(**api_params)
            latency_ms = (time.time() - start_time) * 1000

            # Extract generated text
            generated_text = response.choices[0].message.content

            # Extract usage
            usage = (
                LLMUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )
                if response.usage
                else LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
            )

            # Collect metadata
            metadata = LLMMetadata(
                model=self._model,
                provider="openai",
                latency_ms=latency_ms,
                request_id=response.id,
                timestamp=datetime.now(timezone.utc),
                raw={
                    "model": response.model,
                    "finish_reason": response.choices[0].finish_reason,
                },
            )

            return LLMResponse(content=generated_text, usage=usage, metadata=metadata)
        except Exception as e:
            raise LLMAdapterError(f"Failed to generate text from OpenAI: {str(e)}")

    def stream(
        self,
        prompt: PromptInput,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        user_context: UserContext | None = None,
    ) -> LLMStreamResponse:
        """
        Stream text generation for the given prompt.

        Args:
            prompt: String or list of ChatMessage objects for text generation
            temperature: Sampling temperature (0-2)
            max_output_tokens: Maximum tokens in response
            user_context: User context for the request

        Returns:
            LLMStreamResponse: Generator yielding text chunks and metadata

        Raises:
            LLMAdapterError: If the API call fails
        """
        try:
            start_time = time.time()
            # Convert string prompt to messages format if needed
            messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt

            # Build API parameters
            api_params: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "stream": True,
            }
            if temperature is not None:
                api_params["temperature"] = temperature
            if max_output_tokens is not None:
                api_params["max_tokens"] = max_output_tokens

            stream = self._client.chat.completions.create(**api_params)

            # Extract request_id and usage from stream
            request_id = None
            usage = None

            def chunk_generator() -> Any:
                nonlocal request_id, usage
                for chunk in stream:
                    # Capture request_id from first chunk
                    if request_id is None and hasattr(chunk, "id"):
                        request_id = chunk.id

                    # Capture usage from final chunk
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage = LLMUsage(
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                            total_tokens=chunk.usage.total_tokens,
                        )

                    if chunk.choices[0].delta.content:
                        yield LLMStreamChunk(delta=chunk.choices[0].delta.content)

            latency_ms = (time.time() - start_time) * 1000
            metadata = LLMMetadata(
                model=self._model,
                provider="openai",
                latency_ms=latency_ms,
                request_id=request_id,
                timestamp=datetime.now(timezone.utc),
            )

            return LLMStreamResponse(
                stream=chunk_generator(),
                usage=usage,
                metadata=metadata,
            )
        except Exception as e:
            raise LLMAdapterError(f"Failed to stream from OpenAI: {str(e)}")
