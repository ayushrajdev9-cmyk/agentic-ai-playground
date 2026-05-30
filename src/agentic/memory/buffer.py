from __future__ import annotations

from agentic.memory.base import BaseMemory, MemoryEntry


class BufferMemory(BaseMemory):
    def __init__(self, max_entries: int = 50, max_tokens: int = 4096):
        super().__init__(max_entries)
        self.max_tokens = max_tokens

    def add(self, user_input: str, response: str, **metadata: dict) -> None:
        self.entries.append(MemoryEntry(role="user", content=user_input, metadata=metadata))
        self.entries.append(MemoryEntry(role="assistant", content=response, metadata=metadata))

        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

    def get_context(self, **kwargs) -> str:
        lines = []
        for entry in self.entries:
            prefix = "User" if entry.role == "user" else "Assistant"
            lines.append(f"{prefix}: {entry.content}")
        return "\n".join(lines)

    def get_recent_context(self, n: int = 5) -> str:
        recent = self.get_recent(n * 2)
        lines = []
        for entry in recent:
            prefix = "User" if entry.role == "user" else "Assistant"
            lines.append(f"{prefix}: {entry.content}")
        return "\n".join(lines)
