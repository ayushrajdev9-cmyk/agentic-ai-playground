"""Tool-using agent example with web search and code execution."""
import asyncio
import os

from agentic import ReActAgent, OpenAIChat, WebSearchTool, CodeExecutorTool, BufferMemory


async def main():
    llm = OpenAIChat(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    agent = ReActAgent(
        llm=llm,
        tools=[
            WebSearchTool(max_results=3),
            CodeExecutorTool(timeout=15),
        ],
        memory=BufferMemory(max_entries=10),
    )

    result = await agent.run(
        "Search for the latest Python 3.13 features and write a demo script that showcases one of them."
    )
    print(f"Result:\n{result.output}")
    print(f"\nTotal iterations: {result.total_iterations}")
    print(f"Total tool calls: {result.total_tool_calls}")


if __name__ == "__main__":
    asyncio.run(main())
