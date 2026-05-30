import pytest

from agentic.memory.base import BaseMemory, MemoryEntry
from agentic.memory.buffer import BufferMemory
from agentic.memory.summary import SummaryMemory


class TestMemoryInterface:
    def test_memory_entry_creation(self):
        entry = MemoryEntry(role="user", content="hello")
        assert entry.role == "user"
        assert entry.content == "hello"
        assert entry.metadata == {}

    def test_memory_entry_with_metadata(self):
        entry = MemoryEntry(
            role="assistant",
            content="world",
            metadata={"tokens": 10},
        )
        assert entry.metadata["tokens"] == 10


class TestBufferMemory:
    def test_add_and_get_context(self):
        mem = BufferMemory(max_entries=5)
        mem.add("Hello", "Hi there")
        context = mem.get_context()
        assert "User: Hello" in context
        assert "Assistant: Hi there" in context

    def test_max_entries(self):
        mem = BufferMemory(max_entries=3)
        for i in range(5):
            mem.add(f"msg{i}", f"res{i}")
        context = mem.get_context()
        assert "msg0" not in context
        assert "msg2" in context

    def test_clear(self):
        mem = BufferMemory()
        mem.add("a", "b")
        assert mem.count() == 2
        mem.clear()
        assert mem.count() == 0

    def test_get_recent(self):
        mem = BufferMemory()
        for i in range(10):
            mem.add(f"q{i}", f"a{i}")
        recent = mem.get_recent(4)
        assert len(recent) == 4
        assert recent[0].content == "q6"


class TestSummaryMemory:
    def test_basic_operation(self):
        mem = SummaryMemory(max_entries=10, summary_threshold=20)
        mem.add("Hello", "Hi")
        assert mem.count() == 2
        assert mem.summary is None

    def test_no_summary_without_llm(self):
        mem = SummaryMemory(llm=None, summary_threshold=1)
        mem.add("a", "b")
        mem.add("c", "d")
        assert mem.summary is None
