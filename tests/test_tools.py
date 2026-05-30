import pytest

from agentic.tools.base import BaseTool, ToolResult
from agentic.tools.files import FileOpsTool
from agentic.tools.code import CodeExecutorTool


class MockTool(BaseTool):
    def __init__(self):
        super().__init__()
        self._name = "mock"
        self._description = "A mock tool for testing"

    async def execute(self, **kwargs):
        return ToolResult(output=f"executed with {kwargs}").__str__()


@pytest.mark.asyncio
async def test_base_tool():
    tool = MockTool()
    assert tool.name == "mock"
    assert tool.description == "A mock tool for testing"
    result = await tool.execute(test="value")
    assert "executed" in result


@pytest.mark.asyncio
async def test_file_ops_tool_read_nonexistent(tmp_path):
    tool = FileOpsTool(working_dir=str(tmp_path))
    result = await tool.execute(operation="read", path="/nonexistent/file.txt")
    assert "not found" in result


@pytest.mark.asyncio
async def test_file_ops_tool_write_and_read(tmp_path):
    tool = FileOpsTool(working_dir=str(tmp_path))
    test_file = tmp_path / "test.txt"

    write_result = await tool.execute(
        operation="write",
        path=str(test_file),
        content="Hello, World!",
    )
    assert "Written" in write_result

    read_result = await tool.execute(operation="read", path=str(test_file))
    assert "Hello, World!" in read_result


@pytest.mark.asyncio
async def test_file_ops_tool_list(tmp_path):
    tool = FileOpsTool(working_dir=str(tmp_path))
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "sub").mkdir()

    result = await tool.execute(operation="list", path=str(tmp_path))
    assert "a.txt" in result
    assert "b.txt" in result
    assert "sub/" in result


@pytest.mark.asyncio
async def test_code_executor_python():
    tool = CodeExecutorTool(timeout=10)
    result = await tool.execute(
        code="print(2 + 2)",
        language="python",
    )
    assert "4" in result


@pytest.mark.asyncio
async def test_code_executor_invalid_language():
    tool = CodeExecutorTool(allowed_languages=["python"])
    result = await tool.execute(code="print(1)", language="javascript")
    assert "not allowed" in result


@pytest.mark.asyncio
async def test_code_executor_timeout():
    tool = CodeExecutorTool(timeout=1)
    result = await tool.execute(
        code="import time; time.sleep(10)",
        language="python",
    )
    assert "timed out" in result
