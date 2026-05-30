# Getting Started

## Installation

```bash
# Basic installation
pip install agentic-ai-playground

# With all LLM backends
pip install agentic-ai-playground[all]

# Development setup
git clone https://github.com/yourusername/agentic-ai-playground.git
cd agentic-ai-playground
pip install -e ".[dev]"
```

## Quick Start

### 1. Configure your LLM

```python
from agentic import OpenAIChat

llm = OpenAIChat(model="gpt-4o")
```

Or use local models with Ollama:

```python
from agentic import OllamaChat

llm = OllamaChat(model="llama3")
```

### 2. Create an agent

```python
from agentic import BaseAgent, BufferMemory

agent = BaseAgent(
    llm=llm,
    memory=BufferMemory(),
)
```

### 3. Chat with it

```python
response = await agent.chat("What is the meaning of life?")
print(response)
```

### 4. Add tools

```python
from agentic import ReActAgent, WebSearchTool, CodeExecutorTool

agent = ReActAgent(
    llm=llm,
    tools=[
        WebSearchTool(),
        CodeExecutorTool(),
    ],
    memory=BufferMemory(),
)

result = await agent.run("Find Python tutorials and test one of them")
```

## Configuration

Copy `config/default.yaml` to your project and customize:

```yaml
llm:
  provider: openai
  model: gpt-4o
  temperature: 0.7
```

Pass it when creating agents:

```python
import yaml
from agentic import BaseAgent, OpenAIChat

with open("config.yaml") as f:
    config = yaml.safe_load(f)

llm = OpenAIChat(model=config["llm"]["model"])
agent = BaseAgent(llm=llm)
```
