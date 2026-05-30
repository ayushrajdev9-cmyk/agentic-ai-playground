# Tools

## Creating Tools

All tools extend `BaseTool` and implement `execute()`:

```python
from agentic.tools.base import BaseTool, ToolResult

class WeatherTool(BaseTool):
    def __init__(self):
        super().__init__()
        self._name = "weather"
        self._description = "Get weather for a city"
        self._parameters = {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
            },
            "required": ["city"],
        }

    async def execute(self, **kwargs):
        city = kwargs.get("city", "")
        # call weather API...
        return ToolResult(output=f"Weather in {city}: Sunny, 72°F").__str__()
```

## Built-in Tools

### WebSearchTool
Search the web using DuckDuckGo.

```python
WebSearchTool(max_results=5)
```

### WebFetchTool
Fetch content from a URL.

```python
WebFetchTool()
```

### CodeExecutorTool
Execute code in a sandbox.

```python
CodeExecutorTool(timeout=30, allowed_languages=["python"])
```

### FileOpsTool
Read, write, list, move, copy, delete files.

```python
FileOpsTool(working_dir="/path/to/project")
```

## Tool Result

Every tool returns a string via `ToolResult.__str__()`:

```python
result = ToolResult(
    success=True,
    output="computed value",
    metadata={"key": "value"},
)
str(result)  # -> "computed value"
```
