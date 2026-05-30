"""Run an agent using local Ollama models - no API keys required."""
import asyncio

from agentic import ReActAgent, OllamaChat, WebSearchTool, BufferMemory


async def main():
    llm = OllamaChat(
        model=os.getenv("OLLAMA_MODEL", "llama3"),
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
    )

    agent = ReActAgent(
        llm=llm,
        tools=[WebSearchTool(max_results=3)],
        memory=BufferMemory(max_entries=10),
    )

    result = await agent.run("What are the top programming languages in 2026?")
    print(f"Agent: {result.output}")


if __name__ == "__main__":
    import os
    asyncio.run(main())
