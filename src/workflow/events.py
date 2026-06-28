"""Realtime workflow events for SSE ingress and audit streaming."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field


class WorkflowEventKind(str, Enum):
    WORKFLOW_START = "workflow_start"
    WORKFLOW_STEP = "workflow_step"
    AGENT_START = "agent_start"
    AGENT_COMPLETE = "agent_complete"
    WORKFLOW_COMPLETE = "workflow_complete"
    WORKFLOW_ERROR = "workflow_error"


class WorkflowEvent(BaseModel):
    run_id: str
    kind: WorkflowEventKind
    workflow_step: Optional[str] = None
    agent: Optional[str] = None
    message_type: Optional[str] = None
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_sse(self, *, stream_format: str = "human") -> str:
        """Format as Server-Sent Events. Use stream_format='json' for machine parsing."""
        if stream_format == "json":
            return f"data: {self.model_dump_json()}\n\n"
        from src.workflow.event_display import format_event_human

        text = format_event_human(self)
        if text is None:
            return ""
        lines = text.rstrip("\n").split("\n")
        return "".join(f"data: {line}\n" for line in lines) + "\n"


WorkflowEventCallback = Callable[[WorkflowEvent], None]


def emit_event(
    callback: Optional[WorkflowEventCallback],
    event: WorkflowEvent,
) -> None:
    if callback is not None:
        callback(event)
