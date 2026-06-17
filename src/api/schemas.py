"""HTTP API schemas for MA-RAG ingress."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from src.contracts.messages import FinalAnswerPackage


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    run_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    run_id: str
    question: str
    answer: str
    confidence: int
    route: Optional[str] = None
    verify_passed: Optional[bool] = None
    verify_issues: list[str] = Field(default_factory=list)
    plan_steps: list[str] = Field(default_factory=list)
    workflow_trace: list[str] = Field(default_factory=list)
    chunk_ids_used: list[str] = Field(default_factory=list)
    evidence_ledger_path: Optional[str] = None
    a2a_journal_path: Optional[str] = None

    @classmethod
    def from_package(cls, package: FinalAnswerPackage) -> "AskResponse":
        return cls(
            run_id=package.run_id,
            question=package.question,
            answer=package.answer,
            confidence=package.confidence,
            route=package.route_decision.value if package.route_decision else None,
            verify_passed=package.verify_passed,
            verify_issues=package.verify_issues,
            plan_steps=package.plan_steps,
            workflow_trace=[step.value for step in package.workflow_trace],
            chunk_ids_used=package.chunk_ids_used,
            evidence_ledger_path=package.evidence_ledger_path,
            a2a_journal_path=package.a2a_journal_path,
        )


class HealthResponse(BaseModel):
    status: str
    llm: str
    index_ready: bool


class ToolInvokeRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)
    run_id: Optional[str] = None
    caller_agent: Optional[str] = "api"
