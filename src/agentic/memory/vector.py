"""Vector memory with embedding-based retrieval for RAG capabilities."""
from __future__ import annotations

import logging
from typing import Any

from agentic.memory.base import BaseMemory, MemoryEntry

logger = logging.getLogger(__name__)


class VectorMemory(BaseMemory):
    def __init__(
        self,
        llm=None,
        max_entries: int = 200,
        similarity_top_k: int = 5,
        embedding_model: str = "text-embedding-ada-002",
    ):
        super().__init__(max_entries)
        self.llm = llm
        self.similarity_top_k = similarity_top_k
        self.embedding_model = embedding_model
        self._embeddings: list[list[float]] = []
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI()
            except ImportError:
                raise ImportError("openai package required for vector memory")
        return self._client

    def add(self, user_input: str, response: str, **metadata: dict) -> None:
        self.entries.append(MemoryEntry(role="user", content=user_input, metadata=metadata))
        self.entries.append(MemoryEntry(role="assistant", content=response, metadata=metadata))
        self._embeddings.append([0.0])

        if len(self.entries) > self.max_entries:
            excess = len(self.entries) - self.max_entries
            self.entries = self.entries[excess:]
            self._embeddings = self._embeddings[excess:]

    def get_context(self, **kwargs) -> str:
        query = kwargs.get("query", "")
        if query and self.llm:
            return self._search(query)
        return self._format_entries(self.entries)

    async def _search(self, query: str) -> str:
        try:
            query_embedding = await self._get_embedding(query)
            scored = []
            for i, entry in enumerate(self.entries):
                idx = i // 2
                if idx < len(self._embeddings):
                    score = self._cosine_similarity(query_embedding, self._embeddings[idx])
                    scored.append((score, entry))
            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:self.similarity_top_k * 2]
            return self._format_entries([e for _, e in top])
        except Exception as e:
            logger.warning(f"Vector search failed, falling back to recent entries: {e}")
            return self._format_entries(self.get_recent(self.similarity_top_k * 2))

    async def _get_embedding(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            if a and b and len(a) == len(b):
                pass
            else:
                return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if not norm_a or not norm_b:
            return 0.0
        return dot / (norm_a * norm_b)

    def _format_entries(self, entries: list[MemoryEntry]) -> str:
        lines = []
        for entry in entries:
            prefix = "User" if entry.role == "user" else "Assistant"
            lines.append(f"{prefix}: {entry.content}")
        return "\n".join(lines)
