"""Router / triage agent — classify questions before planning."""

from __future__ import annotations

from src.contracts.messages import RouterRequest, RouterResponse
from src.slm_helpers import classify_route

AGENT_ID = "router"
ROLE = "Router / Triage — route simple RAG vs multi-hop agentic workflow"


class RouterAgent:
    def __init__(self, *, agent_id: str = AGENT_ID, role: str = ROLE):
        self.agent_id = agent_id
        self.role = role

    def run(self, request: RouterRequest) -> RouterResponse:
        decision, rationale = classify_route(request.question)
        return RouterResponse(
            run_id=request.run_id,
            decision=decision,
            rationale=rationale,
        )
