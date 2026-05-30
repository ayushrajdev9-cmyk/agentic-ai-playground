import pytest

from agentic.agents.base import BaseAgent, AgentConfig, AgentResult
from agentic.agents.react import ReActAgent


class MockLLM:
    def __init__(self, responses=None):
        self.responses = responses or ["Final Answer: Hello!"]

    async def generate(self, prompt, **kwargs):
        return self.responses.pop(0) if self.responses else "Final Answer: OK"


class MockMemory:
    def __init__(self):
        self.history = []

    def add(self, user_input, response, **metadata):
        self.history.append((user_input, response))

    def get_context(self, **kwargs):
        return "\n".join(f"User: {u}\nAI: {r}" for u, r in self.history)

    def clear(self):
        self.history.clear()

    def get_recent(self, n=10):
        return []

    def count(self):
        return len(self.history)


@pytest.mark.asyncio
async def test_base_agent_initialization():
    llm = MockLLM()
    agent = BaseAgent(llm=llm)
    assert agent.llm is not None
    assert agent.tools == []
    assert agent.memory is None


@pytest.mark.asyncio
async def test_react_agent_basic():
    llm = MockLLM()
    memory = MockMemory()
    agent = ReActAgent(llm=llm, memory=memory)
    result = await agent.run("Hi")
    assert isinstance(result, AgentResult)
    assert result.output == "Hello!"


@pytest.mark.asyncio
async def test_react_agent_with_memory():
    llm = MockLLM()
    memory = MockMemory()
    agent = ReActAgent(llm=llm, memory=memory)
    await agent.run("First message")
    assert memory.count() == 1
    await agent.run("Second message")
    assert memory.count() == 2


@pytest.mark.asyncio
async def test_react_agent_max_iterations():
    llm = MockLLM(responses=["Thought: testing\nAction: none\nAction Input: {}"])
    agent = ReActAgent(
        llm=llm,
        config=AgentConfig(max_iterations=3),
    )
    result = await agent.run("Loop test")
    assert result.total_iterations == 3
    assert not result.success


def test_agent_config_validation():
    config = AgentConfig(max_iterations=10, verbose=True)
    assert config.max_iterations == 10
    assert config.verbose is True
    assert config.handle_errors == "resilient"

    with pytest.raises(ValueError):
        AgentConfig(max_iterations=0)

    with pytest.raises(ValueError):
        AgentConfig(handle_errors="invalid_option")
