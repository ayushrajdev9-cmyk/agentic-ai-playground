import pytest
from agentic.memory.vector import VectorMemory


@pytest.mark.asyncio
async def test_vector_memory_initialization():
    mem = VectorMemory(max_entries=10)
    assert mem.max_entries == 10
    assert mem.count() == 0
    assert mem.similarity_top_k == 5


@pytest.mark.asyncio
async def test_vector_memory_add_and_clear():
    mem = VectorMemory(max_entries=10)
    mem.add("Hello", "Hi there")
    assert mem.count() == 2
    mem.clear()
    assert mem.count() == 0


@pytest.mark.asyncio
async def test_cosine_similarity():
    mem = VectorMemory()
    assert mem._cosine_similarity([1, 0], [1, 0]) == 1.0
    assert mem._cosine_similarity([1, 0], [0, 1]) == 0.0
    assert mem._cosine_similarity([], []) == 0.0


@pytest.mark.asyncio
async def test_token_tracker():
    from agentic.token_tracker import TokenTracker
    tracker = TokenTracker()
    tracker.record_usage("test", 100, 50, model="gpt-4o")
    summary = tracker.summary()
    assert summary["total_tokens"] == 150
    assert summary["prompt_tokens"] == 100
    assert summary["completion_tokens"] == 50
    assert summary["estimated_cost_usd"] > 0
    assert summary["sessions"] == 1


@pytest.mark.asyncio
async def test_registry():
    from agentic.registry import Registry
    from agentic.agents.base import BaseAgent
    from agentic.tools.base import BaseTool

    registry = Registry()
    registry.clear()

    registry.register_agent("test_agent", BaseAgent)
    registry.register_tool("test_tool", BaseTool)

    assert registry.get_agent("test_agent") == BaseAgent
    assert registry.get_tool("test_tool") == BaseTool
    assert registry.get_agent("nonexistent") is None

    agents = registry.list_agents()
    assert "test_agent" in agents
