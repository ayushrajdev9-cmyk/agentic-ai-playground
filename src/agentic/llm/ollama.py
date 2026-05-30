from __future__ import annotations

import json
import logging
from typing import Any
from collections.abc import AsyncIterator

import httpx

from agentic.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class OllamaChat(BaseLLM):
    def __init__(
        self,
        model: str = "llama3",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        base_url: str = "http://localhost:11434",
    ):
        super().__init__(model, temperature, max_tokens)
        self.base_url = base_url.rstrip("/")

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": kwargs.get("temperature", self.temperature),
                    "options": {"num_predict": kwargs.get("max_tokens", self.max_tokens)},
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")

    async def generate_stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": kwargs.get("temperature", self.temperature),
                    "stream": True,
                },
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                        except json.JSONDecodeError:
                            continue
