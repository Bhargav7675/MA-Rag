"""In-process workflow runtime (enterprise steps 0 → 1 → 4 → 6)."""

from __future__ import annotations

from typing import Any, Optional

from agents.planner_agent import PlannerAgent
from agents.rag_step_agent import RagStepAgent
from agents.step_definer_agent import StepDefinerAgent
from agents.summarizer_agent import SummarizerAgent
from src.contracts.messages import (
    FinalAnswerPackage,
    PlanRequest,
    StepDefineRequest,
    SummarizeRequest,
    UserRequest,
    WorkflowStep,
)
from src.pipeline import normalize_question


class WorkflowEngine:
    """
    Orchestrates typed agents without LangGraph shared state.

    Steps: INIT_PLAN → (per plan step: define → RAG) → FINALIZE
    """

    def __init__(self, retriever_tool: Any):
        self.retriever_tool = retriever_tool
        self.planner = PlannerAgent()
        self.step_definer = StepDefinerAgent()
        self.rag_step = RagStepAgent(retriever_tool)
        self.summarizer = SummarizerAgent()

    def run(self, question: str, *, run_id: Optional[str] = None) -> FinalAnswerPackage:
        question = normalize_question(question)
        user = UserRequest(question=question)
        if run_id:
            user.run_id = run_id

        trace: list[WorkflowStep] = [WorkflowStep.INIT_PLAN]
        plan_res = self.planner.run(
            PlanRequest(run_id=user.run_id, question=user.question)
        )

        step_answers = []
        chunk_ids: list[str] = []
        stop_early = False

        for step_index, plan_step in enumerate(plan_res.steps):
            if stop_early:
                break

            trace.append(WorkflowStep.RETRIEVE)
            define_res = self.step_definer.run(
                StepDefineRequest(
                    run_id=user.run_id,
                    plan_steps=plan_res.steps,
                    current_step_index=step_index,
                    prior_step_answers=step_answers,
                )
            )
            trace.append(WorkflowStep.GENERATE)
            answer = self.rag_step.run(
                run_id=user.run_id,
                step_index=step_index,
                plan_step=plan_step,
                task=define_res.task,
                task_type=define_res.task_type,
            )
            step_answers.append(answer)
            chunk_ids.extend(answer.doc_ids)

            if not answer.success:
                stop_early = True

        trace.append(WorkflowStep.FINALIZE)
        summary = self.summarizer.run(
            SummarizeRequest(
                run_id=user.run_id,
                question=user.question,
                plan_steps=plan_res.steps,
                step_answers=step_answers,
            )
        )

        return FinalAnswerPackage(
            run_id=user.run_id,
            question=user.question,
            answer=summary.answer,
            confidence=summary.confidence,
            plan_steps=plan_res.steps,
            step_answers=step_answers,
            workflow_trace=trace,
            chunk_ids_used=list(dict.fromkeys(chunk_ids)),
        )


def format_workflow_output(package: FinalAnswerPackage) -> str:
    lines: list[str] = []
    lines.append("=== Plan ===")
    for i, step in enumerate(package.plan_steps, start=1):
        lines.append(f"{i}. {step}")

    lines.append("\n=== Step trace ===")
    for step in package.step_answers:
        lines.append(f"\n--- Step {step.step_index + 1} ---")
        lines.append(f"Task: {step.task}")
        lines.append(f"Answer: {step.answer}")
        lines.append(f"Success: {'Yes' if step.success else 'No'}")
        lines.append(f"Confidence: {step.confidence}")

    lines.append("\n=== Final answer ===")
    lines.append(package.answer)
    lines.append(f"Confidence score: {package.confidence}")
    lines.append(f"\nRun id: {package.run_id}")
    lines.append(f"Trace: {', '.join(s.value for s in package.workflow_trace)}")
    return "\n".join(lines)
