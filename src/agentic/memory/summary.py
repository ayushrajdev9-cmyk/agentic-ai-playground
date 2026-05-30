from __future__ import annotations

import logging

from agentic.memory.base import BaseMemory, MemoryEntry

logger = logging.getLogger(__name__)


class SummaryMemory(BaseMemory):
    def __init__(self, llm=None, max_entries: int = 20, summary_threshold: int = 10):
        super().__init__(max_entries)
        self.llm = llm
        self.summary_threshold = summary_threshold
        self.summary: str | None = None

    def add(self, user_input: str, response: str, **metadata: dict) -> None:
        self.entries.append(MemoryEntry(role="user", content=user_input, metadata=metadata))
        self.entries.append(MemoryEntry(role="assistant", content=response, metadata=metadata))

        if len(self.entries) >= self.summary_threshold * 2:
            self._summarize()

        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

    def get_context(self, **kwargs) -> str:
        parts = []
        if self.summary:
            parts.append(f"Conversation Summary:\n{self.summary}\n")
        for entry in self.entries:
            prefix = "User" if entry.role == "user" else "Assistant"
            parts.append(f"{prefix}: {entry.content}")
        return "\n".join(parts)

    async def _summarize(self) -> None:
        if not self.llm:
            return
        text = "\n".join(f"{e.role}: {e.content}" for e in self.entries)
        try:
            summary = await self.llm.generate(
                f"Summarize this conversation concisely:\n\n{text}"
            )
            self.summary = summary
            logger.info(f"Memory summarized: {len(summary)} chars")
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
