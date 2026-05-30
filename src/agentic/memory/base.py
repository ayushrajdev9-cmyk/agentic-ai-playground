from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = Field(default_factory=dict)


class BaseMemory(ABC):
    def __init__(self, max_entries: int = 100):
        self.max_entries = max_entries
        self.entries: list[MemoryEntry] = []

    @abstractmethod
    def add(self, user_input: str, response: str, **metadata: dict) -> None:
        ...

    @abstractmethod
    def get_context(self, **kwargs) -> str:
        ...

    def get_recent(self, n: int = 10) -> list[MemoryEntry]:
        return self.entries[-n:]

    def clear(self) -> None:
        self.entries.clear()

    def count(self) -> int:
        return len(self.entries)
