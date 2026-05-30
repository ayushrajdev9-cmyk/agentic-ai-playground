from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from agentic.llm.base import BaseLLM
from agentic.memory.base import BaseMemory
from agentic.tools.base import BaseTool

logger = logging.getLogger(__name__)


class AgentConfig(BaseModel):
    max_iterations: int = Field(default=25, ge=1)
    max_tool_calls_per_step: int = Field(default=3, ge=1)
    verbose: bool = False
    handle_errors: str = Field(default="resilient", pattern="^(raise|resilient|silent)$")


class AgentStep(BaseModel):
    iteration: int
    thought: str | None = None
    action: str | None = None
    action_input: dict[str, Any] = Field(default_factory=dict)
    observation: str | None = None
    output: str | None = None


class AgentResult(BaseModel):
    output: str
    steps: list[AgentStep] = Field(default_factory=list)
    total_iterations: int = 0
    total_tool_calls: int = 0
    success: bool = True


class BaseAgent(ABC):
    def __init__(
        self,
        llm: BaseLLM,
        tools: list[BaseTool] | None = None,
        memory: BaseMemory | None = None,
        config: AgentConfig | None = None,
    ):
        self.llm = llm
        self.tools = tools or []
        self.memory = memory
        self.config = config or AgentConfig()
        self._tool_map: dict[str, BaseTool] = {t.name: t for t in self.tools}

    def add_tool(self, tool: BaseTool) -> None:
        self.tools.append(tool)
        self._tool_map[tool.name] = tool

    def get_tool(self, name: str) -> BaseTool | None:
        return self._tool_map.get(name)

    @abstractmethod
    async def run(self, prompt: str, **kwargs: Any) -> AgentResult:
        ...

    @abstractmethod
    async def chat(self, message: str, **kwargs: Any) -> str:
        ...

    def reset(self) -> None:
        if self.memory:
            self.memory.clear()

    def tool_descriptions(self) -> str:
        if not self.tools:
            return ""
        lines = ["Available tools:", ""]
        for tool in self.tools:
            lines.append(f"  - {tool.name}: {tool.description}")
            if tool.parameters:
                lines.append(f"    Parameters: {tool.parameters}")
        return "\n".join(lines)
