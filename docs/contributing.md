# Contributing

## Development Setup

```bash
git clone https://github.com/yourusername/agentic-ai-playground.git
cd agentic-ai-playground
make install-dev
```

## Code Style

We use Ruff for linting and formatting:

```bash
make lint
make format
```

## Type Checking

```bash
make typecheck
```

## Testing

```bash
make test
# or
pytest -v
```

## Pull Request Process

1. Fork the repo and create your branch from `main`
2. If adding code, add tests
3. Run `make all` to ensure lint, types, and tests pass
4. Open a PR with a clear description

## Adding a New Tool

1. Create `src/agentic/tools/your_tool.py`
2. Extend `BaseTool` and implement `execute()`
3. Export in `src/agentic/tools/__init__.py`
4. Add tests in `tests/test_tools.py`
5. Document in `docs/tools.md`

## Adding a New LLM Backend

1. Create `src/agentic/llm/your_provider.py`
2. Extend `BaseLLM` and implement `generate()` and `generate_stream()`
3. Export in `src/agentic/llm/__init__.py`
4. Add optional dependency in `pyproject.toml`
