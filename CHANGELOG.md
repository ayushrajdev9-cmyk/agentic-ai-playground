# Changelog

## 0.1.0 (2026-05-30)

### Features
- Modular agent architecture with BaseAgent, ReActAgent, ToolAgent
- Multi-agent orchestration with delegation, parallel execution, and debate
- Research team workflow (researcher -> writer -> critic)
- Pluggable LLM backends for OpenAI and Ollama
- Rich tool system: web search, web fetch, code execution, file operations
- Memory systems: buffer memory and summary memory
- CLI interface for interactive chat, single-run, and API serving
- FastAPI-based REST API server
- Docker and docker-compose support
- YAML configuration system
- Comprehensive test suite

### Infrastructure
- CI/CD with GitHub Actions (lint, typecheck, test)
- Pre-commit hooks for code quality
- Ruff for linting and formatting
- Mypy for type checking
- MIT License
