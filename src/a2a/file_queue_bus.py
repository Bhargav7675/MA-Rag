"""File-queue A2A transport — out-of-process agent workers (Oracle Queue-ready shape)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from src.a2a.bus import InProcessA2ABus
from src.a2a.envelope import A2AEnvelope, A2AResponse, new_message_id
from src.env import get_a2a_queue_dir

if TYPE_CHECKING:
    from src.a2a.file_journal import A2AFileJournal


class FileQueueA2ABus(InProcessA2ABus):
    """
    A2A bus that enqueues requests for remote workers instead of in-process dispatch.

  Local dev: run `python run_a2a_worker.py` in a second terminal.
  Production: swap file dirs for OCI Queue topics without changing envelope shape.
    """

    def __init__(
        self,
        registry=None,
        *,
        journal: Optional["A2AFileJournal"] = None,
        queue_dir: Path | None = None,
        poll_interval_s: float = 0.05,
        request_timeout_s: float = 120.0,
    ):
        super().__init__(registry=registry, journal=journal)
        base = queue_dir or get_a2a_queue_dir()
        self.pending_dir = base / "pending"
        self.response_dir = base / "responses"
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.response_dir.mkdir(parents=True, exist_ok=True)
        self.poll_interval_s = poll_interval_s
        self.request_timeout_s = request_timeout_s

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

        pending_path = self.pending_dir / f"{envelope.message_id}.json"
        pending_path.write_text(envelope.model_dump_json(), encoding="utf-8")

        response_path = self.response_dir / f"{envelope.message_id}.json"
        deadline = time.monotonic() + self.request_timeout_s
        while time.monotonic() < deadline:
            if response_path.exists():
                raw = response_path.read_text(encoding="utf-8")
                response_path.unlink(missing_ok=True)
                response = A2AResponse.model_validate_json(raw)
                self._record(response)
                return response
            time.sleep(self.poll_interval_s)

        pending_path.unlink(missing_ok=True)
        response = A2AResponse(
            message_id=new_message_id(),
            correlation_id=correlation_id,
            from_agent=to_agent,
            to_agent=from_agent,
            message_type=f"{message_type}.response",
            payload={},
            success=False,
            error=f"A2A worker timeout after {self.request_timeout_s}s for {to_agent}",
        )
        self._record(response)
        return response


def process_pending_queue(bus: InProcessA2ABus) -> int:
    """Process one batch of pending A2A envelopes. Returns count handled."""
    if not isinstance(bus, FileQueueA2ABus):
        return 0

    handled = 0
    for pending_path in sorted(bus.pending_dir.glob("*.json")):
        try:
            envelope = A2AEnvelope.model_validate_json(
                pending_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, ValueError):
            pending_path.unlink(missing_ok=True)
            continue

        try:
            result = bus.registry.dispatch(envelope.to_agent, envelope.payload)
            response = A2AResponse(
                message_id=new_message_id(),
                correlation_id=envelope.correlation_id,
                from_agent=envelope.to_agent,
                to_agent=envelope.from_agent,
                message_type=f"{envelope.message_type}.response",
                payload=result,
                success=True,
            )
        except Exception as exc:
            response = A2AResponse(
                message_id=new_message_id(),
                correlation_id=envelope.correlation_id,
                from_agent=envelope.to_agent,
                to_agent=envelope.from_agent,
                message_type=f"{envelope.message_type}.response",
                payload={},
                success=False,
                error=str(exc),
            )

        response_path = bus.response_dir / f"{envelope.message_id}.json"
        response_path.write_text(response.model_dump_json(), encoding="utf-8")
        pending_path.unlink(missing_ok=True)
        handled += 1

    return handled
