"""Enhanced examples showcasing advanced agentic AI patterns."""
import asyncio
import os

from agentic import (
    ReActAgent,
    BaseAgent,
    AgentOrchestrator,
    OpenAIChat,
    WebSearchTool,
    CodeExecutorTool,
    FileOpsTool,
    BufferMemory,
)

SYSTEM_PROMPT = """You are an expert coding assistant that writes production-ready code.
You follow best practices, add appropriate error handling, and write clean code."""


async def code_review_workflow():
    """Multi-agent code review system."""
    llm = OpenAIChat(model=os.getenv("OPENAI_MODEL", "gpt-4o"))

    coder = BaseAgent(llm=llm)
    reviewer = BaseAgent(llm=llm)
    orchestrator = AgentOrchestrator(agents=[coder, reviewer])

    code = """
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)
"""

    results = await orchestrator.debate(
        f"Review this code and suggest improvements:\n```python\n{code}\n```",
        rounds=2,
    )
    return results


async def autonomous_research():
    """Agent that researches, implements, and documents a solution."""
    llm = OpenAIChat(model=os.getenv("OPENAI_MODEL", "gpt-4o"))

    agent = ReActAgent(
        llm=llm,
        tools=[
            WebSearchTool(max_results=5),
            CodeExecutorTool(),
            FileOpsTool(working_dir="/tmp/agent-output"),
        ],
        memory=BufferMemory(),
    )

    result = await agent.run(
        "Research the Fast Fourier Transform algorithm, implement it in Python "
        "with a working example, and save the result."
    )
    return result


async def main():
    print("=== Code Review Workflow ===")
    feedback = await code_review_workflow()
    for i, fb in enumerate(feedback):
        print(f"\nAgent {i + 1}:")
        print(fb[:200] + "...")


if __name__ == "__main__":
    asyncio.run(main())
