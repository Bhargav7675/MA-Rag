"""MCP-style tool gateway for evidence-plane tools."""

from src.tools.faiss_retrieve import FaissRetrieveTool, TOOL_NAME, build_faiss_retrieve_tool
from src.tools.registry import ToolRegistry, default_tool_registry
from src.tools.schemas import ToolCallRequest, ToolCallResult, ToolDefinition

__all__ = [
    "FaissRetrieveTool",
    "TOOL_NAME",
    "ToolCallRequest",
    "ToolCallResult",
    "ToolDefinition",
    "ToolRegistry",
    "build_faiss_retrieve_tool",
    "default_tool_registry",
]
