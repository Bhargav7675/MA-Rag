"""A2A message envelope — typed payloads for agent-to-agent dispatch."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def new_message_id() -> str:
    return uuid4().hex[:16]


class A2AEnvelope(BaseModel):
    """In-process A2A message; future queue/worker adapters can serialize this."""

    message_id: str = Field(default_factory=new_message_id)
    correlation_id: str = Field(
        description="Workflow run_id or parent message_id for tracing."
    )
    from_agent: str
    to_agent: str
    message_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    reply_to: Optional[str] = None


class A2AResponse(BaseModel):
    message_id: str
    correlation_id: str
    from_agent: str
    to_agent: str
    message_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
