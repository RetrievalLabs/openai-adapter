# Contributing to OpenAI Adapter

Thank you for your interest in contributing to the OpenAI Adapter project! We welcome contributions from everyone - whether it's bug reports, feature requests, documentation improvements, or code changes.

**Related Documentation:**
- [README.md](README.md) - Project overview and usage
- [DEVELOPMENT.md](DEVELOPMENT.md) - Setup and development environment

## Getting Started

### For Bug Reports and Feature Requests

1. **Search existing issues** - Check if your issue has already been reported
2. **Create a new issue** - Use the appropriate template and provide as much detail as possible
3. **Include examples** - Code snippets, error messages, and steps to reproduce help us understand better

### For Code Contributions

1. **Fork the repository** - Create your own fork on GitHub
2. **Clone your fork** - `git clone https://github.com/your-username/openai-adapter.git`
3. **Create a branch** - Use a descriptive name: `feature/your-feature` or `fix/your-bug`
4. **Make your changes** - Follow the development guide in DEVELOPMENT.md
5. **Submit a pull request** - Provide a clear description of your changes

## Issues

We welcome all types of issues:

### Bug Reports

Include:
- Description of the issue
- Steps to reproduce
- Expected behavior
- Actual behavior
- Python version and environment
- Error traceback (if applicable)

### Feature Requests

Include:
- Clear description of the feature
- Use case and motivation
- Proposed implementation (optional)
- Examples of how it would be used

### Documentation Issues

- Typos or unclear explanations
- Missing information
- Examples that don't work
- Better ways to explain concepts

## Pull Requests

We review and merge contributions regularly. Here's what we look for:

### Before Submitting

1. **Run quality checks**
   ```bash
   make format    # Format code
   make lint      # Check linting
   make typecheck # Run type checking
   ```

2. **Add docstrings** - All public functions should have docstrings
3. **Update documentation** - If adding features, update README.md
4. **Write clear commit messages** - Use conventional commit format:
   - `feat: add new feature`
   - `fix: resolve bug`
   - `docs: update documentation`
   - `refactor: restructure code`
   - `chore: maintenance tasks`

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring

## Changes Made
- Point 1
- Point 2

## How to Test
Steps to verify the changes work

## Checklist
- [ ] Code formatted with `make format`
- [ ] Linting passes with `make lint`
- [ ] Type checking passes with `make typecheck`
- [ ] Docstrings added/updated
- [ ] Documentation updated (if needed)
```

## Development Workflow

For detailed setup instructions and development tasks, see [DEVELOPMENT.md](DEVELOPMENT.md).

Quick summary:

1. **Setup your environment** - See [DEVELOPMENT.md](DEVELOPMENT.md#setup)
2. **Make your changes** - Implement the feature or fix with proper documentation
3. **Ensure quality** - Run `make format`, `make lint`, and `make typecheck`
4. **Commit and push** - Use conventional commit messages
5. **Create pull request** - GitHub will guide you through the process

## Code Guidelines

### Style

- Follow PEP 8 conventions
- Use meaningful variable and function names
- Keep functions focused and modular
- Add comments for complex logic

### Type Annotations

All functions must have type annotations:

```python
def generate(
    self,
    prompt: PromptInput,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> LLMResponse:
    """Generate text completion for the given prompt."""
    ...
```

### Docstrings

Use Google-style docstrings for all public functions:

```python
def embed(self, query: str) -> QueryEmbeddingResponse:
    """Generate an embedding vector for the given query.

    Args:
        query: The text to embed

    Returns:
        QueryEmbeddingResponse: Contains embedding vector and metadata

    Raises:
        QueryEmbeddingAdapterError: If the API call fails
    """
    ...
```

## Review Process

1. **Automated checks** - Your PR must pass format, lint, and type checks
2. **Code review** - Team members will review your changes
3. **Feedback** - We may request changes or have questions
4. **Approval** - Once approved, the PR will be merged

## Areas for Contribution

We're particularly interested in:

- **Bug fixes** - Found an issue? Help us fix it!
- **Documentation** - Clearer explanations, examples, translations
- **Performance improvements** - Optimize existing code
- **New features** - Propose and implement useful additions
- **Tests** - Improve test coverage
- **Error handling** - Better error messages and edge case handling

## Community Guidelines

- **Be respectful** - Treat everyone with courtesy
- **Be constructive** - Provide helpful feedback
- **Be patient** - Review takes time
- **Ask questions** - If something is unclear, ask
- **Share knowledge** - Help others understand the codebase

## Getting Help

- **Documentation** - Check README.md and DEVELOPMENT.md
- **Existing issues** - Search for similar problems
- **Discussions** - Open a GitHub discussion for questions
- **Contact** - Reach out if you need guidance

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (RetrievalLabs Business-Restricted License v1.0).

## Recognition

Contributors are recognized in:
- Commit history
- GitHub contributors page
- Release notes (for significant contributions)

## Questions?

- Check existing documentation ([README.md](README.md), [DEVELOPMENT.md](DEVELOPMENT.md))
- Search open/closed issues
- Create a new discussion
- Open an issue with your question

## Documentation

- **[README.md](README.md)** - Installation, quick start, API reference, and configuration
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Setting up development environment and tools
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - This file - contribution guidelines

Thank you for making OpenAI Adapter better! 🎉
