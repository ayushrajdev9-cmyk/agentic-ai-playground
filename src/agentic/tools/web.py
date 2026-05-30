from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from agentic.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    def __init__(self, max_results: int = 5):
        super().__init__()
        self._name = "web_search"
        self._description = "Search the web for information. Returns a list of relevant results with titles and snippets."
        self._parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                },
            },
            "required": ["query"],
        }
        self.max_results = max_results

    async def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", self.max_results)

        if not query:
            return ToolResult(success=False, error="Query is required").__str__()

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.get(
                    "https://api.duckduckgo.com",
                    params={
                        "q": query,
                        "format": "json",
                        "max_results": max_results,
                    },
                )
                response.raise_for_status()
                data = response.json()
                results = data.get("results", [])
                if not results:
                    results = data.get("relatedTopics", [])
                formatted = []
                for r in results[:max_results]:
                    title = r.get("title", r.get("Text", ""))
                    snippet = r.get("snippet", r.get("Text", ""))
                    url = r.get("url", "")
                    formatted.append(f"- {title}: {snippet} ({url})")
                return ToolResult(
                    output="\n".join(formatted) or "No results found.",
                    metadata={"query": query, "count": len(formatted)},
                ).__str__()
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return ToolResult(success=False, error=str(e)).__str__()


class WebFetchTool(BaseTool):
    def __init__(self):
        super().__init__()
        self._name = "web_fetch"
        self._description = "Fetch and extract text content from a URL."
        self._parameters = {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return",
                },
            },
            "required": ["url"],
        }

    async def execute(self, **kwargs: Any) -> str:
        url = kwargs.get("url", "")
        max_chars = kwargs.get("max_chars", 5000)

        if not url:
            return ToolResult(success=False, error="URL is required").__str__()

        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                response = await client.get(url, headers={"User-Agent": "AgenticAI/1.0"})
                response.raise_for_status()
                text = response.text[:max_chars]
                return ToolResult(
                    output=text,
                    metadata={"url": url, "chars": len(text)},
                ).__str__()
        except Exception as e:
            logger.error(f"Web fetch failed for {url}: {e}")
            return ToolResult(success=False, error=str(e)).__str__()
