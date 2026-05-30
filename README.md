# Agentic AI Playground

Build, test, and deploy AI agents with modular tooling, memory systems, and LLM integrations.

[![CI](https://github.com/yourusername/agentic-ai-playground/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/agentic-ai-playground/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Features

- **Modular Agent Architecture** — ReAct, Tool-using, and custom agent patterns
- **Pluggable LLM Backends** — OpenAI, Ollama, Anthropic, and custom providers
- **Rich Tool System** — Web search, code execution, file operations, API calls
- **Memory Systems** — Buffer, summary, vector, and persistent memory
- **Multi-Agent Orchestration** — Delegate, debate, and collaborate patterns
- **Observability** — Built-in logging, tracing, and metrics

## Quick Start

```bash
pip install agentic-ai-playground

# or from source
git clone https://github.com/yourusername/agentic-ai-playground.git
cd agentic-ai-playground
pip install -e ".[dev]"
```

```python
from agentic import Agent, OpenAIChat

agent = Agent(
    llm=OpenAIChat(model="gpt-4o"),
    tools=["web_search", "code_executor"],
    memory="buffer"
)

response = agent.run("Search for the latest AI papers and summarize them")
print(response)
```

## Architecture

```
agentic-ai-playground/
├── src/agentic/
│   ├── agents/          # Agent implementations
│   │   ├── base.py      # Abstract base agent
│   │   ├── react.py     # ReAct reasoning agent
│   │   └── tool.py      # Tool-using agent
│   ├── tools/           # Tool implementations
│   │   ├── base.py      # Tool base class
│   │   ├── web.py       # Web search & fetch
│   │   ├── code.py      # Code execution sandbox
│   │   └── files.py     # File operations
│   ├── memory/          # Memory systems
│   │   ├── base.py      # Memory interface
│   │   ├── buffer.py    # Conversation buffer
│   │   └── summary.py   # Summarizing memory
│   └── llm/             # LLM backends
│       ├── base.py      # LLM interface
│       ├── openai.py    # OpenAI provider
│       └── ollama.py    # Local Ollama provider
├── examples/            # Usage examples
├── tests/               # Test suite
└── docs/                # Documentation
```

## Examples

### Basic Chat Agent

```python
from agentic import Agent, OllamaChat

agent = Agent(
    llm=OllamaChat(model="llama3"),
    memory="buffer"
)

agent.chat("What is agentic AI?")
```

### Multi-Agent Research System

```python
from agentic import ResearchTeam

team = ResearchTeam(
    researcher=Agent(llm=llm, tools=["web_search", "fetch"]),
    writer=Agent(llm=llm, tools=["file_ops"]),
    critic=Agent(llm=llm)
)

report = team.run("Latest breakthroughs in AI agents")
```

## Configuration

```yaml
# config/default.yaml
llm:
  provider: openai
  model: gpt-4o
  temperature: 0.7

memory:
  type: buffer
  max_tokens: 4096

tools:
  - web_search
  - code_executor
  - file_ops

logging:
  level: INFO
  format: structured
```

## Roadmap

- [x] Base agent architecture
- [x] ReAct agent pattern
- [x] Tool system
- [x] Memory management
- [x] Multi-agent orchestration
- [ ] Streaming responses
- [ ] Vector memory (RAG)
- [ ] MCP protocol support
- [ ] WebUI dashboard
- [ ] Docker deployment

## Contributing

See [CONTRIBUTING.md](docs/contributing.md) for guidelines.

## License

MIT
