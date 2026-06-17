"""MCP-style tool contracts (schema + call + result)."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    name: str
    type: Literal["string", "integer", "number", "boolean", "object", "array"]
    description: str = ""
    required: bool = True
    default: Optional[Any] = None


class ToolDefinition(BaseModel):
    """Descriptor exposed to agents and HTTP /tools (MCP-inspired)."""

    name: str
    description: str
    parameters: list[ToolParameter] = Field(default_factory=list)
    version: str = "1.0.0"


class ToolCallRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    run_id: Optional[str] = None
    caller_agent: Optional[str] = None


class RetrievedChunkResult(BaseModel):
    doc_id: str
    text: str
    score: Optional[float] = None
    source: Optional[str] = None


class ToolCallResult(BaseModel):
    tool_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None
