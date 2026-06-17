"""Registry of MCP-style tools available to agents."""

from __future__ import annotations

from typing import Any, Callable, Optional

from src.tools.faiss_retrieve import FaissRetrieveTool, TOOL_NAME, build_faiss_retrieve_tool
from src.tools.schemas import ToolCallRequest, ToolCallResult, ToolDefinition


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Any] = {}
        self._definitions: dict[str, ToolDefinition] = {}

    def register(self, tool: Any) -> None:
        if not hasattr(tool, "definition") or not hasattr(tool, "invoke"):
            raise TypeError("Tool must expose .definition and .invoke()")
        name = tool.definition.name
        self._tools[name] = tool
        self._definitions[name] = tool.definition

    def register_factory(self, name: str, factory: Callable[[], Any]) -> None:
        self.register(factory())

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._definitions.values())

    def get_definition(self, name: str) -> ToolDefinition:
        if name not in self._definitions:
            raise KeyError(f"Unknown tool: {name}")
        return self._definitions[name]

    def invoke(self, request: ToolCallRequest) -> ToolCallResult:
        tool = self._tools.get(request.tool_name)
        if tool is None:
            return ToolCallResult(
                tool_name=request.tool_name,
                success=False,
                error=f"Unknown tool: {request.tool_name}",
            )
        return tool.invoke(request)


def default_tool_registry(retriever: Any, *, top_k: int = 3) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(build_faiss_retrieve_tool(retriever, top_k=top_k))
    return registry
