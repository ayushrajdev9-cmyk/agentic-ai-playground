"""FastAPI-based API server for Agentic AI Playground."""
from __future__ import annotations

import logging
import os
from typing import Any

from agentic import BaseAgent, OpenAIChat, BufferMemory
from agentic.config import Config

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    FastAPI = None
    BaseModel = None
    uvicorn = None


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    temperature: float | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


class RunRequest(BaseModel):
    prompt: str
    max_iterations: int | None = None


class RunResponse(BaseModel):
    output: str
    total_iterations: int
    total_tool_calls: int
    success: bool


def create_app(config_path: str | None = None) -> Any:
    if FastAPI is None:
        raise ImportError("fastapi and uvicorn are required. Install with: pip install fastapi uvicorn")

    app = FastAPI(
        title="Agentic AI Playground API",
        version="0.1.0",
        description="API for interacting with AI agents",
    )

    config = Config(config_path) if config_path else Config()
    sessions: dict[str, BaseAgent] = {}

    def get_agent(session_id: str | None = None) -> tuple[BaseAgent, str]:
        sid = session_id or "default"
        if sid not in sessions:
            llm = OpenAIChat(
                model=config.get("llm.model", "gpt-4o"),
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            sessions[sid] = BaseAgent(llm=llm, memory=BufferMemory())
        return sessions[sid], sid

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        agent, sid = get_agent(request.session_id)
        try:
            response = await agent.chat(
                request.message,
                temperature=request.temperature,
            )
            return ChatResponse(response=response, session_id=sid)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/run", response_model=RunResponse)
    async def run(request: RunRequest):
        agent, _ = get_agent()
        try:
            result = await agent.run(request.prompt)
            return RunResponse(
                output=result.output,
                total_iterations=result.total_iterations,
                total_tool_calls=result.total_tool_calls,
                success=result.success,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/sessions/{session_id}")
    async def clear_session(session_id: str):
        if session_id in sessions:
            sessions[session_id].reset()
            del sessions[session_id]
        return {"status": "cleared"}

    return app


async def serve(host: str = "0.0.0.0", port: int = 8000, config_path: str | None = None):
    app = create_app(config_path)
    logger.info(f"Starting API server on {host}:{port}")
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    import asyncio
    asyncio.run(serve())
