from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    success: bool = True
    output: str = ""
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        if self.error:
            return f"Error: {self.error}"
        return self.output


class BaseTool(ABC):
    def __init__(self):
        self._name: str = self.__class__.__name__.replace("Tool", "").lower()
        self._description: str = ""
        self._parameters: dict = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict:
        return self._parameters

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        ...

    def validate_input(self, **kwargs: Any) -> bool:
        return True
