# OpenAI Adapter for RAG Control

A high-performance OpenAI adapter for RAG (Retrieval-Augmented Generation) systems, providing seamless integration with rag-control framework for LLM and embedding operations.

## Features

- 🚀 **High-Performance LLM Generation** - Support for chat completions with streaming
- 🔍 **Query Embeddings** - Generate embeddings for text queries using OpenAI's embedding models
- 📊 **Comprehensive Metadata** - Track latency, token usage, and request IDs
- ⚙️ **Flexible Configuration** - Full OpenAI client configuration support
- 🛡️ **Error Handling** - Robust exception handling with custom adapter errors
- 📝 **Type-Safe** - Full type annotations for Python static analysis

## Installation

Install the package from PyPI:

```bash
pip install openai-adapter
```

Or with uv:

```bash
uv pip install openai-adapter
```

For development installation from source:

```bash
git clone <repo-url>
cd openai-adapter
uv sync
```

### Version Compatibility

- **Current Stable Version:** v0.1.0
- **Compatible with:** rag-control v0.1.3

## Quick Start

### LLM Adapter

Generate text completions using OpenAI's language models:

```python
from openai_adapter import OpenAILLMAdapter

# Initialize adapter
adapter = OpenAILLMAdapter(api_key="sk-...")

# Simple text generation
response = adapter.generate("What is machine learning?")
print(response.text)
print(response.metadata)  # latency, usage, model info

# With parameters
response = adapter.generate(
    "What is machine learning?",
    temperature=0.7,
    max_output_tokens=100
)

# Streaming
stream_response = adapter.stream("Tell me a story")
for chunk in stream_response.stream:
    print(chunk.text, end="", flush=True)
print(stream_response.metadata)  # Final metadata with usage
```

### Query Embedding Adapter

Generate embeddings for text queries:

```python
from openai_adapter import OpenAIQueryEmbeddingAdapter

# Initialize adapter
adapter = OpenAIQueryEmbeddingAdapter(api_key="sk-...")

# Generate embedding
response = adapter.embed("machine learning algorithms")
print(response.embedding)  # Vector of embeddings
print(response.metadata)   # Model, dimensions, latency
```

## API Reference

### OpenAILLMAdapter

#### `__init__(api_key: str, model: str = "gpt-3.5-turbo", **kwargs)`

Initialize the LLM adapter.

**Parameters:**
- `api_key` (str): OpenAI API key for authentication
- `model` (str): Language model to use (default: gpt-3.5-turbo)
- `**kwargs`: Additional OpenAI client configuration
  - `organization` (str): Organization ID
  - `project` (str): Project ID
  - `base_url` (str): Custom API endpoint
  - `timeout` (float): Request timeout in seconds

**Raises:**
- `LLMAdapterError`: If client initialization fails

#### `generate(prompt, temperature=None, max_output_tokens=None)`

Generate a text completion.

**Parameters:**
- `prompt` (str | list[dict]): Text or list of chat messages
- `temperature` (float, optional): Sampling temperature (0-2)
- `max_output_tokens` (int, optional): Maximum tokens in response

**Returns:**
- `LLMResponse`: Contains generated text and metadata

#### `stream(prompt, temperature=None, max_output_tokens=None)`

Stream text generation in real-time.

**Parameters:**
- Same as `generate()`

**Returns:**
- `LLMStreamResponse`: Generator yielding text chunks with final metadata

### OpenAIQueryEmbeddingAdapter

#### `__init__(api_key: str, model: str = "text-embedding-3-small", **kwargs)`

Initialize the embedding adapter.

**Parameters:**
- `api_key` (str): OpenAI API key
- `model` (str): Embedding model (default: text-embedding-3-small)
- `**kwargs`: Additional OpenAI client configuration

**Raises:**
- `QueryEmbeddingAdapterError`: If client initialization fails

#### `embed(query: str)`

Generate an embedding vector.

**Parameters:**
- `query` (str): Text to embed

**Returns:**
- `QueryEmbeddingResponse`: Contains embedding vector and metadata

## Configuration

### Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
```

### Custom Endpoints

```python
adapter = OpenAILLMAdapter(
    api_key="sk-...",
    base_url="https://custom-endpoint.com/v1"
)
```

### Organization & Project

```python
adapter = OpenAILLMAdapter(
    api_key="sk-...",
    organization="org-123",
    project="proj-456"
)
```

## Metadata

All responses include comprehensive metadata:

```python
response = adapter.generate("prompt")

# Access metadata
metadata = response.metadata
print(metadata.model)              # "gpt-3.5-turbo"
print(metadata.provider)           # "openai"
print(metadata.latency_ms)         # API latency in milliseconds
print(metadata.request_id)         # OpenAI request ID
print(metadata.timestamp)          # Request timestamp (UTC)
print(metadata.usage.prompt_tokens)        # Input tokens
print(metadata.usage.completion_tokens)    # Output tokens
print(metadata.usage.total_tokens)         # Total tokens
print(metadata.raw)                # Raw API response data
```

## Error Handling

```python
from rag_control.exceptions import LLMAdapterError, QueryEmbeddingAdapterError

try:
    response = adapter.generate("prompt")
except LLMAdapterError as e:
    print(f"LLM Error: {e}")
except QueryEmbeddingAdapterError as e:
    print(f"Embedding Error: {e}")
```

## Supported Models

### LLM Models
- `gpt-4o` - Latest multimodal model
- `gpt-4-turbo` - High-performance model
- `gpt-3.5-turbo` - Fast and cost-effective (default)

### Embedding Models
- `text-embedding-3-large` - Most powerful
- `text-embedding-3-small` - Fast and efficient (default)

## Development & Contributing

For information on setting up a development environment, code style guidelines, and contribution guidelines, see:

- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Development environment setup and code quality tools
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contributing guidelines and pull request process

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) file for details.

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review error messages for guidance
