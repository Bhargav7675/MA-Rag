"""Register workflow agents on the in-process A2A bus."""

from __future__ import annotations

from typing import Any

from agents.critic_agent import CriticAgent
from agents.evidence_curator_agent import EvidenceCuratorAgent
from agents.planner_agent import PlannerAgent
from agents.retrieval_agent import RetrievalAgent
from agents.router_agent import RouterAgent
from agents.step_definer_agent import StepDefinerAgent
from agents.summarizer_agent import SummarizerAgent
from src.a2a.bus import InProcessA2ABus
from src.a2a.registry import AgentDescriptor
from src.contracts.messages import (
    EvidenceReviewRequest,
    PlanRequest,
    RetrievalTask,
    RouterRequest,
    StepDefineRequest,
    SummarizeRequest,
    VerifyRequest,
)


def setup_a2a_bus(
    *,
    router: RouterAgent,
    planner: PlannerAgent,
    retrieval: RetrievalAgent,
    evidence_curator: EvidenceCuratorAgent,
    step_definer: StepDefinerAgent,
    summarizer: SummarizerAgent,
    critic: CriticAgent,
) -> InProcessA2ABus:
    bus = InProcessA2ABus()

    bus.register_agent(
        AgentDescriptor(
            agent_id="router",
            role=router.role,
            message_types=["router.request"],
        ),
        lambda payload: router.run(RouterRequest(**payload)).model_dump(),
    )
    bus.register_agent(
        AgentDescriptor(
            agent_id="planner",
            role=planner.role,
            message_types=["plan.request"],
        ),
        lambda payload: planner.run(PlanRequest(**payload)).model_dump(),
    )
    bus.register_agent(
        AgentDescriptor(
            agent_id="retrieval",
            role=retrieval.role,
            message_types=["retrieval.task"],
            description="FAISS retrieval via MCP-style faiss_retrieve tool",
        ),
        lambda payload: retrieval.run(RetrievalTask(**payload)).model_dump(),
    )
    bus.register_agent(
        AgentDescriptor(
            agent_id="evidence_curator",
            role=evidence_curator.role,
            message_types=["evidence.review"],
        ),
        lambda payload: evidence_curator.run(EvidenceReviewRequest(**payload)).model_dump(),
    )
    bus.register_agent(
        AgentDescriptor(
            agent_id="step_definer",
            role=step_definer.role,
            message_types=["step.define"],
        ),
        lambda payload: step_definer.run(StepDefineRequest(**payload)).model_dump(),
    )
    bus.register_agent(
        AgentDescriptor(
            agent_id="summarizer",
            role=summarizer.role,
            message_types=["summarize.request"],
        ),
        lambda payload: summarizer.run(SummarizeRequest(**payload)).model_dump(),
    )
    bus.register_agent(
        AgentDescriptor(
            agent_id="critic",
            role=critic.role,
            message_types=["verify.request"],
        ),
        lambda payload: critic.run(VerifyRequest(**payload)).model_dump(),
    )

    return bus


def a2a_request(
    bus: InProcessA2ABus,
    *,
    to_agent: str,
    message_type: str,
    payload: dict[str, Any],
    correlation_id: str,
    from_agent: str = "workflow",
) -> dict[str, Any]:
    response = bus.request(
        from_agent=from_agent,
        to_agent=to_agent,
        message_type=message_type,
        payload=payload,
        correlation_id=correlation_id,
    )
    if not response.success:
        raise RuntimeError(response.error or f"A2A request to {to_agent} failed")
    return response.payload
