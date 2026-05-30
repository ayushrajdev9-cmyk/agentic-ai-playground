"""Command-line interface for the Agentic AI Playground."""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from agentic import (
    BaseAgent,
    ReActAgent,
    OpenAIChat,
    OllamaChat,
    WebSearchTool,
    CodeExecutorTool,
    BufferMemory,
)
from agentic.config import Config
from agentic.utils import setup_logging


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic",
        description="Agentic AI Playground - Build and interact with AI agents",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--model", "-m",
        default=os.getenv("OPENAI_MODEL", "gpt-4o"),
        help="LLM model to use",
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["openai", "ollama"],
        default="openai",
        help="LLM provider",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        help="Ollama base URL",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )

    subparsers = parser.add_subparsers(dest="command")

    chat_parser = subparsers.add_parser("chat", help="Interactive chat with an agent")
    chat_parser.add_argument("--tools", "-t", nargs="*", default=[],
                             choices=["web_search", "code_executor", "file_ops"],
                             help="Tools to enable")

    run_parser = subparsers.add_parser("run", help="Run a single prompt")
    run_parser.add_argument("prompt", nargs="+", help="The prompt to run")
    run_parser.add_argument("--tools", "-t", nargs="*", default=[],
                            choices=["web_search", "code_executor", "file_ops"],
                            help="Tools to enable")

    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    serve_parser.add_argument("--port", "-p", type=int, default=8000, help="Port to bind to")

    return parser


def build_agent(args):
    if args.provider == "ollama":
        llm = OllamaChat(model=args.model, base_url=args.ollama_url)
    else:
        llm = OpenAIChat(
            model=args.model,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    tools = []
    if hasattr(args, "tools"):
        tool_map = {
            "web_search": WebSearchTool,
            "code_executor": CodeExecutorTool,
            "file_ops": lambda: FileOpsTool(),
        }
        for t_name in args.tools or []:
            if t_name in tool_map:
                tools.append(tool_map[t_name]())

    memory = BufferMemory()
    if tools:
        return ReActAgent(llm=llm, tools=tools, memory=memory)
    return BaseAgent(llm=llm, memory=memory)


async def cmd_chat(args):
    agent = build_agent(args)
    print("🤖 Agentic AI Chat (type 'exit' to quit)")
    print("─" * 50)
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ("exit", "quit"):
                break
            if not user_input:
                continue
            response = await agent.chat(user_input)
            print(f"\nAgent: {response}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


async def cmd_run(args):
    agent = build_agent(args)
    prompt = " ".join(args.prompt)
    result = await agent.run(prompt) if isinstance(agent, ReActAgent) else await agent.chat(prompt)
    if isinstance(result, str):
        print(result)
    else:
        print(result.output)
        if args.verbose:
            print(f"\nSteps: {result.total_iterations}, Tool calls: {result.total_tool_calls}")


async def cmd_serve(args):
    try:
        from agentic.api import serve
        await serve(host=args.host, port=args.port)
    except ImportError:
        print("FastAPI is required for the server. Install with: pip install fastapi uvicorn")
        sys.exit(1)


async def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.config:
        config = Config(args.config)
        setup_logging(config.get("logging.level", "INFO"))
    else:
        setup_logging("INFO" if not args.verbose else "DEBUG")

    if args.command == "chat":
        await cmd_chat(args)
    elif args.command == "run":
        await cmd_run(args)
    elif args.command == "serve":
        await cmd_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
