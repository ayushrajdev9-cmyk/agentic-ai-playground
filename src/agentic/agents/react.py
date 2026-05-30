from __future__ import annotations

import json
import logging
import re
from typing import Any

from agentic.agents.base import BaseAgent, AgentResult, AgentStep

logger = logging.getLogger(__name__)

THOUGHT_PATTERN = re.compile(r"Thought:\s*(.*?)(?=Action:|Final Answer:|$)", re.DOTALL)
ACTION_PATTERN = re.compile(r"Action:\s*(\w+)\s*\nAction Input:\s*(.*?)(?=Observation:|$)", re.DOTALL)
FINAL_PATTERN = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)

REACT_PROMPT_TEMPLATE = """You are a helpful AI assistant that uses structured reasoning and tools.

You operate in a loop of: Thought, Action, Action Input, Observation.

When you have enough information, respond with: Final Answer: <your answer>

{tool_descriptions}

Use the following format:

Thought: What do I need to do next?
Action: tool_name
Action Input: {{"param": "value"}}
Observation: result of the action
... (repeat as needed)
Final Answer: The final answer to the user's question

Now begin.

{chat_history}
User: {prompt}
Thought:"""


class ReActAgent(BaseAgent):
    async def run(self, prompt: str, **kwargs: Any) -> AgentResult:
        steps: list[AgentStep] = []
        total_tool_calls = 0

        chat_history = self.memory.get_context() if self.memory else ""

        for iteration in range(self.config.max_iterations):
            agent_prompt = REACT_PROMPT_TEMPLATE.format(
                tool_descriptions=self.tool_descriptions(),
                chat_history=chat_history,
                prompt=prompt,
            )

            try:
                response = await self.llm.generate(agent_prompt)
            except Exception as e:
                logger.error(f"LLM generation failed at iteration {iteration}: {e}")
                if self.config.handle_errors == "raise":
                    raise
                return AgentResult(
                    output=f"Error: {e}",
                    steps=steps,
                    total_iterations=iteration + 1,
                    total_tool_calls=total_tool_calls,
                    success=False,
                )

            final_match = FINAL_PATTERN.search(response)
            if final_match:
                final_answer = final_match.group(1).strip()
                steps.append(AgentStep(
                    iteration=iteration,
                    output=final_answer,
                ))
                if self.memory:
                    self.memory.add(prompt, final_answer)
                return AgentResult(
                    output=final_answer,
                    steps=steps,
                    total_iterations=iteration + 1,
                    total_tool_calls=total_tool_calls,
                )

            thought_match = THOUGHT_PATTERN.search(response)
            action_match = ACTION_PATTERN.search(response)

            thought = thought_match.group(1).strip() if thought_match else ""
            action_name = action_match.group(1).strip() if action_match else ""
            action_input_str = action_match.group(2).strip() if action_match else ""

            if not action_name:
                steps.append(AgentStep(
                    iteration=iteration,
                    thought=thought,
                    output=response,
                ))
                if self.memory:
                    self.memory.add(prompt, response)
                return AgentResult(
                    output=response,
                    steps=steps,
                    total_iterations=iteration + 1,
                    total_tool_calls=total_tool_calls,
                )

            try:
                action_input = json.loads(action_input_str) if action_input_str else {}
            except json.JSONDecodeError:
                action_input = {"input": action_input_str}

            tool = self.get_tool(action_name)
            if not tool:
                observation = f"Error: Unknown tool '{action_name}'. Available: {list(self._tool_map.keys())}"
            else:
                try:
                    total_tool_calls += 1
                    observation = await tool.execute(**action_input)
                except Exception as e:
                    observation = f"Tool execution error: {e}"

            step = AgentStep(
                iteration=iteration,
                thought=thought,
                action=action_name,
                action_input=action_input,
                observation=observation,
            )
            steps.append(step)
            chat_history += f"\nThought: {thought}\nAction: {action_name}\nAction Input: {action_input_str}\nObservation: {observation}\n"

            if self.config.verbose:
                logger.info(f"Iteration {iteration}: {action_name}({action_input}) -> {observation[:100]}...")

        return AgentResult(
            output="Max iterations reached without final answer.",
            steps=steps,
            total_iterations=self.config.max_iterations,
            total_tool_calls=total_tool_calls,
            success=False,
        )

    async def chat(self, message: str, **kwargs: Any) -> str:
        result = await self.run(message, **kwargs)
        return result.output
