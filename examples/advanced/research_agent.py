"""Research team: researcher + writer + critic workflow."""
import asyncio
import os

from agentic import (
    ReActAgent,
    BaseAgent,
    ResearchTeam,
    OpenAIChat,
    WebSearchTool,
    FileOpsTool,
    BufferMemory,
)


async def main():
    llm = OpenAIChat(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    team = ResearchTeam(
        researcher=ReActAgent(
            llm=llm,
            tools=[WebSearchTool(max_results=8)],
            memory=BufferMemory(),
        ),
        writer=BaseAgent(
            llm=llm,
            memory=BufferMemory(),
        ),
        critic=BaseAgent(
            llm=llm,
            memory=BufferMemory(),
        ),
    )

    report = await team.run("The rise of agentic AI in 2026: trends, tools, and impact.")
    print(f"Final Report:\n\n{report.output}")


if __name__ == "__main__":
    asyncio.run(main())
