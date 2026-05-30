"""Multi-agent system demonstrating delegation and parallel execution."""
import asyncio
import os

from agentic import (
    BaseAgent,
    ReActAgent,
    AgentOrchestrator,
    OpenAIChat,
    WebSearchTool,
    CodeExecutorTool,
    BufferMemory,
)


async def main():
    llm = OpenAIChat(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    researcher = ReActAgent(
        llm=llm,
        tools=[WebSearchTool(max_results=5)],
        memory=BufferMemory(),
    )

    coder = ReActAgent(
        llm=llm,
        tools=[CodeExecutorTool()],
        memory=BufferMemory(),
    )

    coordinator = BaseAgent(
        llm=llm,
        memory=BufferMemory(),
    )

    orchestrator = AgentOrchestrator(
        agents=[researcher, coder],
        coordinator=coordinator,
    )

    print("=== Parallel Research ===")
    results = await orchestrator.parallel_run(
        "Research and implement a solution for sorting 1 million integers efficiently."
    )
    for i, result in enumerate(results):
        print(f"\n--- Agent {i} ---")
        print(result.output[:300] + "...")


if __name__ == "__main__":
    asyncio.run(main())
