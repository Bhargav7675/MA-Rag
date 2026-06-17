"""In-process A2A message bus (scaffolding for future queue/worker runtime)."""

from __future__ import annotations

from typing import Any, Callable, Optional, TYPE_CHECKING

from src.a2a.envelope import A2AEnvelope, A2AResponse, new_message_id
from src.a2a.registry import AgentDescriptor, AgentRegistry

if TYPE_CHECKING:
    from src.a2a.file_journal import A2AFileJournal


class InProcessA2ABus:
    """
    Synchronous request/response bus for typed agent messages.

    Today: same-process handlers. Tomorrow: swap transport (Redis, OCI queue, etc.)
    without changing envelope shape or agent contracts.
    """

    def __init__(
        self,
        registry: Optional[AgentRegistry] = None,
        *,
        journal: Optional["A2AFileJournal"] = None,
    ):
        self.registry = registry or AgentRegistry()
        self.journal = journal
        self._history: list[A2AEnvelope | A2AResponse] = []
        self._max_history = 500

    def register_agent(
        self,
        descriptor: AgentDescriptor,
        handler: Callable[[dict], dict],
    ) -> None:
        self.registry.register(descriptor, handler)

    def request(
        self,
        *,
        from_agent: str,
        to_agent: str,
        message_type: str,
        payload: dict[str, Any],
        correlation_id: str,
    ) -> A2AResponse:
        envelope = A2AEnvelope(
            message_id=new_message_id(),
            correlation_id=correlation_id,
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,
            payload=payload,
        )
        self._record(envelope)

        try:
            result = self.registry.dispatch(to_agent, payload)
            response = A2AResponse(
                message_id=new_message_id(),
                correlation_id=correlation_id,
                from_agent=to_agent,
                to_agent=from_agent,
                message_type=f"{message_type}.response",
                payload=result,
                success=True,
            )
        except Exception as exc:
            response = A2AResponse(
                message_id=new_message_id(),
                correlation_id=correlation_id,
                from_agent=to_agent,
                to_agent=from_agent,
                message_type=f"{message_type}.response",
                payload={},
                success=False,
                error=str(exc),
            )

        self._record(response)
        return response

    def list_agents(self) -> list[AgentDescriptor]:
        return self.registry.list_agents()

    def recent_messages(self, limit: int = 50) -> list[A2AEnvelope | A2AResponse]:
        return self._history[-limit:]

    def _record(self, item: A2AEnvelope | A2AResponse) -> None:
        self._history.append(item)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]
        if self.journal is not None:
            correlation_id = item.correlation_id
            record = item.model_dump() if hasattr(item, "model_dump") else dict(item)
            self.journal.append(correlation_id, record)
