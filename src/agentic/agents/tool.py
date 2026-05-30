from __future__ import annotations

import json
import logging
from typing import Any

from agentic.agents.base import BaseAgent, AgentResult, AgentStep

logger = logging.getLogger(__name__)

TOOL_PROMPT_TEMPLATE = """You are an AI assistant with access to tools. For each user request, decide which tools to use and in what order.

{tool_descriptions}

Respond with a JSON object:
{{
  "thought": "your reasoning",
  "tool_calls": [
    {{"tool": "tool_name", "input": {{"param": "value"}}}}
  ]
}}

User: {prompt}
"""


class ToolAgent(BaseAgent):
    async def run(self, prompt: str, **kwargs: Any) -> AgentResult:
        steps: list[AgentStep] = []
        total_tool_calls = 0

        agent_prompt = TOOL_PROMPT_TEMPLATE.format(
            tool_descriptions=self.tool_descriptions(),
            prompt=prompt,
        )

        response = await self.llm.generate(agent_prompt)

        try:
            parsed = json.loads(response)
            thought = parsed.get("thought", "")
            tool_calls = parsed.get("tool_calls", [])

            observations = []
            for tc in tool_calls:
                tool_name = tc.get("tool", "")
                tool_input = tc.get("input", {})
                tool = self.get_tool(tool_name)

                if not tool:
                    obs = f"Unknown tool: {tool_name}"
                else:
                    obs = await tool.execute(**tool_input)
                    total_tool_calls += 1

                observations.append({"tool": tool_name, "result": obs})

                step = AgentStep(
                    iteration=0,
                    thought=thought,
                    action=tool_name,
                    action_input=tool_input,
                    observation=obs,
                )
                steps.append(step)

            summary_prompt = f"Given the original request: {prompt}\n\nTool results:\n{json.dumps(observations, indent=2)}\n\nProvide a final answer."
            final = await self.llm.generate(summary_prompt)

            if self.memory:
                self.memory.add(prompt, final)

            return AgentResult(
                output=final,
                steps=steps,
                total_iterations=1,
                total_tool_calls=total_tool_calls,
            )

        except json.JSONDecodeError:
            return AgentResult(
                output=response,
                steps=steps,
                total_iterations=1,
                total_tool_calls=total_tool_calls,
                success=False,
            )

    async def chat(self, message: str, **kwargs: Any) -> str:
        result = await self.run(message, **kwargs)
        return result.output
