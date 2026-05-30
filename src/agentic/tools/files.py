from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agentic.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class FileOpsTool(BaseTool):
    def __init__(self, working_dir: str | None = None):
        super().__init__()
        self._name = "file_ops"
        self._description = "Read, write, list, and manage files in the filesystem."
        self._parameters = {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "File operation to perform",
                    "enum": ["read", "write", "list", "delete", "copy", "move"],
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (for write operation)",
                },
                "destination": {
                    "type": "string",
                    "description": "Destination path (for copy/move operations)",
                },
            },
            "required": ["operation", "path"],
        }
        self.working_dir = Path(working_dir or ".").resolve()

    def _resolve_path(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.working_dir / p
        return p.resolve()

    async def execute(self, **kwargs: Any) -> str:
        operation = kwargs.get("operation", "")
        path = kwargs.get("path", "")
        content = kwargs.get("content", "")
        destination = kwargs.get("destination", "")

        if not operation or not path:
            return ToolResult(success=False, error="Both 'operation' and 'path' are required").__str__()

        resolved = self._resolve_path(path)

        try:
            if operation == "read":
                if not resolved.exists():
                    return ToolResult(success=False, error=f"File not found: {resolved}").__str__()
                text = resolved.read_text(encoding="utf-8")
                return ToolResult(output=text, metadata={"path": str(resolved), "chars": len(text)}).__str__()

            elif operation == "write":
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.write_text(content, encoding="utf-8")
                return ToolResult(output=f"Written {len(content)} bytes to {resolved}").__str__()

            elif operation == "list":
                if not resolved.exists():
                    return ToolResult(success=False, error=f"Directory not found: {resolved}").__str__()
                items = []
                for item in sorted(resolved.iterdir()):
                    suffix = "/" if item.is_dir() else ""
                    items.append(f"{item.name}{suffix}")
                return ToolResult(output="\n".join(items), metadata={"count": len(items)}).__str__()

            elif operation == "delete":
                if not resolved.exists():
                    return ToolResult(success=False, error=f"Path not found: {resolved}").__str__()
                if resolved.is_file():
                    resolved.unlink()
                else:
                    import shutil
                    shutil.rmtree(resolved)
                return ToolResult(output=f"Deleted {resolved}").__str__()

            elif operation in ("copy", "move"):
                if not resolved.exists():
                    return ToolResult(success=False, error=f"Source not found: {resolved}").__str__()
                dest = self._resolve_path(destination)
                if operation == "copy":
                    import shutil
                    if resolved.is_file():
                        shutil.copy2(resolved, dest)
                    else:
                        shutil.copytree(resolved, dest)
                else:
                    resolved.rename(dest)
                return ToolResult(output=f"{operation.capitalize()}d {resolved} -> {dest}").__str__()

            else:
                return ToolResult(success=False, error=f"Unknown operation: {operation}").__str__()

        except Exception as e:
            logger.error(f"File operation failed: {e}")
            return ToolResult(success=False, error=str(e)).__str__()
