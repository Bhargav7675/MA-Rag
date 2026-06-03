"""
Typed request/response contracts for in-process and future A2A agents.

Ingress (SSO, API gateway, tenants) is out of scope — UserRequest is CLI/API-shaped only.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def new_run_id() -> str:
    return uuid4().hex[:12]


class WorkflowStep(str, Enum):
    """Enterprise workflow runtime steps (subset used in Track B)."""

    INIT_PLAN = "init_plan"
    RETRIEVE = "retrieve"
    EVIDENCE_CHECK = "evidence_check"
    CONTEXT_BUILD = "context_build"
    GENERATE = "generate"
    VERIFY = "verify"
    FINALIZE = "finalize"
    CLARIFY = "clarify"
    ESCALATE = "escalate"


class AgentRole(str, Enum):
    ROUTER = "router"
    PLANNER = "planner"
    RETRIEVAL = "retrieval"
    EVIDENCE_CURATOR = "evidence_curator"
    QA = "qa"
    CRITIC = "critic"
    SUMMARIZER = "summarizer"


class RouteDecision(str, Enum):
    SIMPLE_RAG = "simple_rag"
    MULTI_HOP_RAG = "multi_hop_rag"
    AGGREGATE_ONLY = "aggregate_only"


class StepTaskType(str, Enum):
    QUESTION_ANSWERING = "question-answering"
    AGGREGATE = "aggregate"


class EvidenceSufficiency(str, Enum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


# --- Ingress-shaped input (no auth / tenant plane) ---


class UserRequest(BaseModel):
    """Question entry from CLI or a future API — not a full ingress UserRequest."""

    run_id: str = Field(default_factory=new_run_id)
    question: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# --- Router (Track C; defined now for contract stability) ---


class RouterRequest(BaseModel):
    run_id: str
    question: str


class RouterResponse(BaseModel):
    run_id: str
    decision: RouteDecision
    rationale: str = ""


# --- Planner / Supervisor ---


class PlanRequest(BaseModel):
    run_id: str
    question: str
    past_trial_summaries: list[str] = Field(
        default_factory=list,
        description="Serialized outcomes from prior plan attempts",
    )


class PlanResponse(BaseModel):
    run_id: str
    analysis: str
    steps: list[str]


# --- Step-level answer (shared by definer, verify, summarize) ---


class StepAnswer(BaseModel):
    step_index: int
    plan_step: str
    task: str
    analysis: str = ""
    answer: str
    success: bool
    confidence: int = Field(ge=0, le=10)
    doc_ids: list[str] = Field(default_factory=list)


# --- Step definer ---


class StepDefineRequest(BaseModel):
    run_id: str
    plan_steps: list[str]
    current_step_index: int
    prior_step_answers: list[StepAnswer] = Field(default_factory=list)


class StepDefineResponse(BaseModel):
    run_id: str
    step_index: int
    task_type: StepTaskType
    task: str


# --- Retrieval ---


class RetrievalTask(BaseModel):
    run_id: str
    step_index: int = 0
    question: str
    top_k: int = 3


class RetrievedChunk(BaseModel):
    doc_id: str
    text: str
    score: Optional[float] = None


class RetrievalResponse(BaseModel):
    run_id: str
    step_index: int
    question: str
    chunks: list[RetrievedChunk]


# --- Evidence curator (Track C) ---


class EvidenceReviewRequest(BaseModel):
    run_id: str
    step_index: int
    question: str
    chunks: list[RetrievedChunk]
    extract_notes: list[str] = Field(default_factory=list)


class EvidenceReviewResponse(BaseModel):
    run_id: str
    step_index: int
    sufficiency: EvidenceSufficiency
    gaps: list[str] = Field(default_factory=list)
    proceed: bool = True
    rationale: str = ""


# --- Critic / verify (Track C) ---


class VerifyRequest(BaseModel):
    run_id: str
    question: str
    draft_answer: str
    step_answers: list[StepAnswer] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)


class VerifyResponse(BaseModel):
    run_id: str
    passed: bool
    confidence: int = Field(ge=0, le=10)
    issues: list[str] = Field(default_factory=list)
    revised_answer: Optional[str] = None


# --- Summarizer / finalize ---


class SummarizeRequest(BaseModel):
    run_id: str
    question: str
    plan_steps: list[str]
    step_answers: list[StepAnswer]


class SummarizeResponse(BaseModel):
    run_id: str
    answer: str
    confidence: int = Field(ge=0, le=10)
    summary: str = ""


class FinalAnswerPackage(BaseModel):
    """Deliverable after workflow step FINALIZE."""

    run_id: str
    question: str
    answer: str
    confidence: int
    plan_steps: list[str]
    step_answers: list[StepAnswer]
    workflow_trace: list[WorkflowStep] = Field(default_factory=list)
    chunk_ids_used: list[str] = Field(default_factory=list)
