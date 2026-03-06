# Development Guide

This guide covers how to set up a development environment for the openai-adapter project.

**Related Documentation:**
- [README.md](README.md) - Project overview and usage
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contributing guidelines and pull request process

## Setup

### Prerequisites

- Python 3.9+
- Git
- uv (recommended)

### Development Environment

Clone the repository and set up the development environment:

```bash
# Clone the repository
git clone <repo-url>
cd openai-adapter

# Create virtual environment
make venv

# Activate virtual environment
source .venv/bin/activate

# Install development dependencies
make sync-dev
```

## Code Quality

### Formatting and Linting

Format code and check style:

```bash
# Format code (ruff + black)
make format

# Check code style
make lint
```

### Type Checking

Type safety is enforced with `mypy`:

```bash
# Run type checking
make typecheck
```

**Note:** Some rag-control imports may show warnings about missing type stubs - this is expected and not an error.

## Common Development Tasks

### Adding a New Feature

1. Create a feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```

2. Implement the feature

3. Format and lint:
   ```bash
   make format
   make lint
   ```

4. Type check:
   ```bash
   make typecheck
   ```

5. Commit and push:
   ```bash
   git add .
   git commit -m "feat: add my feature"
   git push origin feature/my-feature
   ```

### Fixing a Bug

1. Create a bug fix branch:
   ```bash
   git checkout -b fix/bug-name
   ```

2. Implement the fix

3. Follow the same formatting/linting steps

### Running All Quality Checks

```bash
# Format code
make format

# Lint code
make lint

# Type check
make typecheck
```

## Project Structure

```
openai-adapter/
├── openai_adapter/
│   ├── __init__.py                 # Package exports
│   ├── llm/
│   │   ├── __init__.py            # LLM module exports
│   │   └── adapter.py             # OpenAILLMAdapter implementation
│   ├── query_embedding/
│   │   ├── __init__.py            # Query embedding exports
│   │   └── adapter.py             # OpenAIQueryEmbeddingAdapter
│   └── version.py                 # Version info
├── pyproject.toml                 # Project configuration
├── README.md                       # User documentation
├── DEVELOPMENT.md                 # This file
└── Makefile                        # Development commands
```

## Documentation

### Docstring Format

Follow Google-style docstrings:

```python
def generate(
    self,
    prompt: PromptInput,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> LLMResponse:
    """Generate text completion for the given prompt.

    Args:
        prompt: String or list of ChatMessage objects
        temperature: Sampling temperature (0-2)
        max_output_tokens: Maximum tokens in response

    Returns:
        LLMResponse: Contains generated text and metadata

    Raises:
        LLMAdapterError: If the API call fails

    Example:
        >>> adapter = OpenAILLMAdapter(api_key="sk-...")
        >>> response = adapter.generate("hello")
        >>> print(response.text)
    """
```

### Type Annotations

Always use type annotations for function parameters and return types:

```python
def generate(
    self,
    prompt: PromptInput,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> LLMResponse:
    ...
```

## Release Process

1. Update version in `openai_adapter/version.py`
2. Update CHANGELOG.md with changes
3. Commit version bump:
   ```bash
   git commit -m "chore: bump version to X.Y.Z"
   ```
4. Tag the release:
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

## Troubleshooting

### Type Checking Errors from rag-control

If you see errors like:
```
error: Skipping analyzing "rag_control.adapters": module is installed, but missing library stubs or py.typed marker
```

This is expected - rag-control doesn't provide type stubs. These are not errors in our code.

### Import Errors

Ensure you've set up the environment:

```bash
make venv
make sync-dev
source .venv/bin/activate
```

### Make Command Issues

If make commands fail, ensure uv is installed and available:

```bash
pip install uv
```

## Code Review & Contributing

For detailed guidelines on contributing to this project, including pull request process, code review, and community guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

When submitting code for review:

1. Verify type checking passes: `make typecheck`
2. Confirm code is formatted: `make format`
3. Confirm linting passes: `make lint`
4. Add docstrings to new functions
5. Update README.md if adding new features
6. Include clear commit messages

## Available Make Commands

- `make venv` - Create virtual environment
- `make activate` - Show activation command
- `make sync-dev` - Install development dependencies
- `make format` - Format code with ruff and black
- `make lint` - Check code with ruff
- `make typecheck` - Run mypy type checking
- `make clean` - Remove generated files

## References

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [rag-control Documentation](https://github.com/RetrievalLabs/rag-control)
- [ruff Documentation](https://docs.astral.sh/ruff/)
- [mypy Documentation](https://mypy.readthedocs.io/)
