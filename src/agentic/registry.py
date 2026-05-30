"""Central registry for managing and discovering agents, tools, and LLMs."""
from __future__ import annotations

from typing import Any

from agentic.agents.base import BaseAgent
from agentic.tools.base import BaseTool
from agentic.llm.base import BaseLLM


class Registry:
    _instance: Registry | None = None

    def __new__(cls) -> Registry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._agents: dict[str, type[BaseAgent]] = {}
            cls._instance._tools: dict[str, type[BaseTool]] = {}
            cls._instance._llms: dict[str, type[BaseLLM]] = {}
            cls._instance._instances: dict[str, Any] = {}
        return cls._instance

    def register_agent(self, name: str, agent_cls: type[BaseAgent]) -> None:
        self._agents[name] = agent_cls

    def register_tool(self, name: str, tool_cls: type[BaseTool]) -> None:
        self._tools[name] = tool_cls

    def register_llm(self, name: str, llm_cls: type[BaseLLM]) -> None:
        self._llms[name] = llm_cls

    def get_agent(self, name: str) -> type[BaseAgent] | None:
        return self._agents.get(name)

    def get_tool(self, name: str) -> type[BaseTool] | None:
        return self._tools.get(name)

    def get_llm(self, name: str) -> type[BaseLLM] | None:
        return self._llms.get(name)

    def list_agents(self) -> list[str]:
        return list(self._agents.keys())

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def list_llms(self) -> list[str]:
        return list(self._llms.keys())

    def register_instance(self, name: str, instance: Any) -> None:
        self._instances[name] = instance

    def get_instance(self, name: str) -> Any:
        return self._instances.get(name)

    def clear(self) -> None:
        self._agents.clear()
        self._tools.clear()
        self._llms.clear()
        self._instances.clear()


registry = Registry()
