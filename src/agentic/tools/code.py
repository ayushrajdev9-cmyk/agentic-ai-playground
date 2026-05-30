from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from agentic.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class CodeExecutorTool(BaseTool):
    def __init__(self, timeout: int = 30, allowed_languages: list[str] | None = None):
        super().__init__()
        self._name = "code_executor"
        self._description = "Execute code in a sandboxed environment. Supports python, javascript, bash."
        self._parameters = {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code to execute",
                },
                "language": {
                    "type": "string",
                    "description": "Programming language (python, javascript, bash)",
                    "enum": ["python", "javascript", "bash"],
                },
            },
            "required": ["code", "language"],
        }
        self.timeout = timeout
        self.allowed_languages = allowed_languages or ["python", "javascript", "bash"]

    async def execute(self, **kwargs: Any) -> str:
        code = kwargs.get("code", "")
        language = kwargs.get("language", "python")

        if language not in self.allowed_languages:
            return ToolResult(
                success=False,
                error=f"Language '{language}' not allowed. Allowed: {self.allowed_languages}",
            ).__str__()

        ext_map = {"python": ".py", "javascript": ".js", "bash": ".sh"}
        cmd_map = {"python": ["python3"], "javascript": ["node"], "bash": ["bash"]}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=ext_map.get(language, ".txt"), delete=False
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [*cmd_map[language], tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            if result.returncode != 0:
                return ToolResult(
                    success=False,
                    output=output,
                    error=f"Exit code: {result.returncode}",
                    metadata={"language": language, "exit_code": result.returncode},
                ).__str__()
            return ToolResult(
                output=output or "(no output)",
                metadata={"language": language, "exit_code": 0},
            ).__str__()
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"Execution timed out after {self.timeout}s",
            ).__str__()
        except FileNotFoundError as e:
            return ToolResult(
                success=False,
                error=f"Runtime not found: {e}. Is {language} installed?",
            ).__str__()
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
            ).__str__()
        finally:
            Path(tmp_path).unlink(missing_ok=True)
