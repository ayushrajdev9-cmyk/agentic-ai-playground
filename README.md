# Agentic AI Playground 🤖

**The fastest way to build, test, and deploy AI agents.** ReAct patterns, multi-agent orchestration, pluggable LLMs, and a rich tool system — all in one modular framework.

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-yellow?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/agents-ReAct%20%7C%20Tool%20%7C%20Custom-brightgreen?style=flat-square" alt="Agent Types">
  <img src="https://img.shields.io/badge/LLM-OpenAI%20%7C%20Ollama%20%7C%20Any-purple?style=flat-square" alt="LLM Support">
</p>

```bash
pip install agentic-ai-playground

# Or from source
git clone https://github.com/ayushrajdev9-cmyk/agentic-ai-playground.git
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

## Demo

```python
from agentic import ReActAgent, OpenAIChat, WebSearchTool, BufferMemory

agent = ReActAgent(
    llm=OpenAIChat(model="gpt-4o"),
    tools=[WebSearchTool()],
    memory=BufferMemory(),
)

# Single query
result = await agent.run("Research the latest AI agent frameworks and compare them")

# Interactive chat
response = await agent.chat("What did we find about LangChain vs CrewAI?")
```

Or from the CLI:
```bash
agentic chat --tools web_search
agentic run "Explain quantum computing in simple terms" --tools web_search
agentic serve --port 8000
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
