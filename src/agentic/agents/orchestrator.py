from __future__ import annotations

import asyncio
import logging
from typing import Any

from agentic.agents.base import BaseAgent, AgentResult

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    def __init__(self, agents: list[BaseAgent], coordinator: BaseAgent | None = None):
        self.agents = agents
        self.coordinator = coordinator or agents[0]

    async def delegate(self, prompt: str, agent_index: int | None = None) -> list[AgentResult]:
        if agent_index is not None:
            target = self.agents[agent_index]
            return [await target.run(prompt)]
        return [await agent.run(prompt) for agent in self.agents]

    async def parallel_run(self, prompt: str) -> list[AgentResult]:
        tasks = [agent.run(prompt) for agent in self.agents]
        return await asyncio.gather(*tasks)

    async def debate(self, prompt: str, rounds: int = 2) -> list[str]:
        opinions = [await agent.run(prompt) for agent in self.agents]
        for _ in range(rounds - 1):
            critique_prompt = prompt + "\n\nOther agents said:\n"
            for i, result in enumerate(opinions):
                critique_prompt += f"\nAgent {i}: {result.output}\n"
            critique_prompt += "\nRefine your response considering the above."
            opinions = [await agent.run(critique_prompt) for agent in self.agents]
        return [r.output for r in opinions]


class ResearchTeam:
    def __init__(
        self,
        researcher: BaseAgent,
        writer: BaseAgent,
        critic: BaseAgent | None = None,
    ):
        self.researcher = researcher
        self.writer = writer
        self.critic = critic

    async def run(self, topic: str) -> AgentResult:
        research = await self.researcher.run(f"Research this topic thoroughly: {topic}")
        draft = await self.writer.run(
            f"Write a comprehensive report based on this research:\n{research.output}"
        )
        if self.critic:
            review = await self.critic.run(
                f"Review and improve this report. Return the final version:\n{draft.output}"
            )
            return review
        return draft
