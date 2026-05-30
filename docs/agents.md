# Agents

## BaseAgent

The foundational agent class. It holds an LLM, optional tools, and optional memory.

```python
from agentic import BaseAgent, OpenAIChat

agent = BaseAgent(
    llm=OpenAIChat(model="gpt-4o"),
    tools=[...],
    memory=...,
)
```

## ReActAgent

Implements the Reasoning + Acting (ReAct) pattern. The agent thinks, acts, observes, and repeats until it has a final answer.

```python
from agentic import ReActAgent

agent = ReActAgent(llm=llm, tools=[...], memory=...)
result = await agent.run("Research and summarize...")
```

**How it works:**
1. Generate a thought about what to do next
2. Choose a tool and provide input
3. Observe the result
4. Repeat until a Final Answer is produced
5. Capped by `max_iterations` in AgentConfig

## ToolAgent

A simpler agent that plans all tool calls upfront in JSON format, executes them, then summarizes.

```python
from agentic import ToolAgent

agent = ToolAgent(llm=llm, tools=[...])
result = await agent.run("Do X, Y, and Z")
```

Best for tasks where the full plan is clear upfront.

## AgentOrchestrator

Manages multiple agents. Supports delegation, parallel execution, and debate.

```python
from agentic import AgentOrchestrator

orchestrator = AgentOrchestrator(agents=[agent1, agent2])

# Run all agents on the same task in parallel
results = await orchestrator.parallel_run("Some task")

# Let agents debate and refine
opinions = await orchestrator.debate("Controversial topic")
```

## ResearchTeam

A specialized multi-agent workflow: researcher -> writer -> critic.

```python
from agentic import ResearchTeam

team = ResearchTeam(
    researcher=research_agent,
    writer=write_agent,
    critic=critic_agent,
)

report = await team.run("Topic to research")
```

## Custom Agents

Subclass `BaseAgent` and implement `run()` and `chat()`:

```python
from agentic import BaseAgent, AgentResult

class MyCustomAgent(BaseAgent):
    async def run(self, prompt, **kwargs):
        response = await self.llm.generate(prompt)
        return AgentResult(output=response)

    async def chat(self, message, **kwargs):
        result = await self.run(message)
        return result.output
```
