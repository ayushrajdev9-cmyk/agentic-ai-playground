"""Basic chat agent example using Ollama or OpenAI."""
import asyncio
import os

from agentic import BaseAgent, OpenAIChat, OllamaChat, BufferMemory


async def main():
    llm = OpenAIChat(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    agent = BaseAgent(
        llm=llm,
        memory=BufferMemory(max_entries=20),
    )

    response = await agent.chat("What is agentic AI and why is it trending in 2026?")
    print(f"Agent: {response}")


if __name__ == "__main__":
    asyncio.run(main())
