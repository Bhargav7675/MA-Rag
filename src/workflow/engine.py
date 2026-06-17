"""In-process workflow runtime (enterprise steps 0 → 1 → 2 → 4 → 5 → 6)."""

from __future__ import annotations

from typing import Any, Optional

from agents.critic_agent import CriticAgent
from agents.evidence_curator_agent import EvidenceCuratorAgent
from agents.planner_agent import PlannerAgent
from agents.rag_step_agent import RagStepAgent
from agents.retrieval_agent import RetrievalAgent
from agents.router_agent import RouterAgent
from agents.step_definer_agent import StepDefinerAgent
from agents.summarizer_agent import SummarizerAgent
from src.contracts.messages import (
    EvidenceReviewRequest,
    FinalAnswerPackage,
    PlanRequest,
    RetrievalTask,
    RouterRequest,
    StepAnswer,
    StepDefineRequest,
    StepTaskType,
    SummarizeRequest,
    UserRequest,
    VerifyRequest,
    WorkflowStep,
)
from src.evidence_ledger import EvidenceLedger
from src.pipeline import normalize_question
from src.workflow.finalize_helpers import (
    all_steps_successful,
    format_kb_multi_hop_answer,
    mean_step_confidence,
    reconcile_final_answer,
)


class WorkflowEngine:
    """
    Orchestrates typed agents without LangGraph shared state.

    Steps: ROUTE → INIT_PLAN → (per plan step: RETRIEVE → EVIDENCE_CHECK → GENERATE)
           → FINALIZE draft → VERIFY → FINALIZE
    """

    def __init__(self, retriever_tool: Any):
        self.retriever_tool = retriever_tool
        self.router = RouterAgent()
        self.planner = PlannerAgent()
        self.retrieval = RetrievalAgent(retriever_tool)
        self.evidence_curator = EvidenceCuratorAgent()
        self.step_definer = StepDefinerAgent()
        self.rag_step = RagStepAgent(retriever_tool)
        self.summarizer = SummarizerAgent()
        self.critic = CriticAgent()

    def run(self, question: str, *, run_id: Optional[str] = None) -> FinalAnswerPackage:
        question = normalize_question(question)
        user = UserRequest(question=question)
        if run_id:
            user.run_id = run_id

        ledger = EvidenceLedger(user.run_id)
        trace: list[WorkflowStep] = [WorkflowStep.ROUTE]
        route_res = self.router.run(
            RouterRequest(run_id=user.run_id, question=user.question)
        )
        ledger.append(
            agent="router",
            workflow_step=WorkflowStep.ROUTE.value,
            payload={
                "decision": route_res.decision.value,
                "rationale": route_res.rationale,
            },
        )

        trace.append(WorkflowStep.INIT_PLAN)
        plan_res = self.planner.run(
            PlanRequest(
                run_id=user.run_id,
                question=user.question,
                route_decision=route_res.decision,
            )
        )
        ledger.append(
            agent="planner",
            workflow_step=WorkflowStep.INIT_PLAN.value,
            payload={"steps": plan_res.steps, "analysis": plan_res.analysis},
        )

        step_answers: list[StepAnswer] = []
        chunk_ids: list[str] = []
        stop_early = False

        for step_index, plan_step in enumerate(plan_res.steps):
            if stop_early:
                break

            if len(plan_res.steps) == 1:
                task = user.question
                task_type = StepTaskType.QUESTION_ANSWERING
            else:
                define_res = self.step_definer.run(
                    StepDefineRequest(
                        run_id=user.run_id,
                        plan_steps=plan_res.steps,
                        current_step_index=step_index,
                        prior_step_answers=step_answers,
                    )
                )
                task = define_res.task
                task_type = define_res.task_type

            if task_type == StepTaskType.AGGREGATE and not step_answers:
                task_type = StepTaskType.QUESTION_ANSWERING
                task = task or plan_step

            if task_type == StepTaskType.AGGREGATE:
                trace.append(WorkflowStep.GENERATE)
                answer = self.rag_step.run(
                    run_id=user.run_id,
                    step_index=step_index,
                    plan_step=plan_step,
                    task=task,
                    task_type=task_type,
                )
                step_answers.append(answer)
                ledger.append(
                    agent="rag_step",
                    workflow_step=WorkflowStep.GENERATE.value,
                    payload={
                        "step_index": step_index,
                        "task": task,
                        "task_type": task_type.value,
                        "answer": answer.answer,
                        "success": answer.success,
                    },
                )
                if not answer.success and len(plan_res.steps) > 1:
                    stop_early = True
                continue

            trace.append(WorkflowStep.RETRIEVE)
            retrieval = self.retrieval.run(
                RetrievalTask(
                    run_id=user.run_id,
                    step_index=step_index,
                    question=task,
                )
            )
            chunk_ids.extend(chunk.doc_id for chunk in retrieval.chunks)
            ledger.append(
                agent="retrieval",
                workflow_step=WorkflowStep.RETRIEVE.value,
                payload={
                    "step_index": step_index,
                    "question": task,
                    "chunk_ids": [chunk.doc_id for chunk in retrieval.chunks],
                },
            )

            trace.append(WorkflowStep.EVIDENCE_CHECK)
            evidence = self.evidence_curator.run(
                EvidenceReviewRequest(
                    run_id=user.run_id,
                    step_index=step_index,
                    question=task,
                    chunks=retrieval.chunks,
                )
            )
            ledger.append(
                agent="evidence_curator",
                workflow_step=WorkflowStep.EVIDENCE_CHECK.value,
                payload={
                    "step_index": step_index,
                    "sufficiency": evidence.sufficiency.value,
                    "proceed": evidence.proceed,
                    "gaps": evidence.gaps,
                    "rationale": evidence.rationale,
                },
            )

            if not evidence.proceed:
                step_answers.append(
                    StepAnswer(
                        step_index=step_index,
                        plan_step=plan_step,
                        task=task,
                        analysis=evidence.rationale,
                        answer="",
                        success=False,
                        confidence=0,
                        doc_ids=[chunk.doc_id for chunk in retrieval.chunks],
                    )
                )
                if len(plan_res.steps) > 1:
                    stop_early = True
                continue

            trace.append(WorkflowStep.CONTEXT_BUILD)
            trace.append(WorkflowStep.GENERATE)
            answer = self.rag_step.run(
                run_id=user.run_id,
                step_index=step_index,
                plan_step=plan_step,
                task=task,
                task_type=task_type,
                documents=[chunk.text for chunk in retrieval.chunks],
                doc_ids=[chunk.doc_id for chunk in retrieval.chunks],
            )
            step_answers.append(answer)
            ledger.append(
                agent="rag_step",
                workflow_step=WorkflowStep.GENERATE.value,
                payload={
                    "step_index": step_index,
                    "task": task,
                    "answer": answer.answer,
                    "success": answer.success,
                    "confidence": answer.confidence,
                },
            )

            if not answer.success and len(plan_res.steps) > 1:
                stop_early = True

        trace.append(WorkflowStep.FINALIZE)
        single_success = (
            len(plan_res.steps) == 1
            and len(step_answers) == 1
            and step_answers[0].success
            and step_answers[0].answer.strip()
        )

        if single_success:
            # Avoid SLM summarize/critic overwriting a correct grounded step answer
            # (e.g. confusing "completed Phase 0" with "next Phase 1" in context).
            step = step_answers[0]
            final_answer = step.answer
            final_confidence = step.confidence
            verify_passed = True
            verify_issues: list[str] = []
            ledger.append(
                agent="summarizer",
                workflow_step=WorkflowStep.FINALIZE.value,
                payload={
                    "draft_answer": final_answer,
                    "confidence": final_confidence,
                    "mode": "single_step_pass_through",
                },
            )
            trace.append(WorkflowStep.VERIFY)
            ledger.append(
                agent="critic",
                workflow_step=WorkflowStep.VERIFY.value,
                payload={
                    "passed": True,
                    "confidence": final_confidence,
                    "issues": [],
                    "revised_answer": final_answer,
                    "mode": "single_step_pass_through",
                },
            )
        elif all_steps_successful(step_answers) and len(step_answers) >= 2:
            formatted = format_kb_multi_hop_answer(step_answers)
            if formatted:
                final_answer = formatted
                final_confidence = mean_step_confidence(step_answers)
                verify_passed = True
                verify_issues = []
                ledger.append(
                    agent="summarizer",
                    workflow_step=WorkflowStep.FINALIZE.value,
                    payload={
                        "draft_answer": final_answer,
                        "confidence": final_confidence,
                        "mode": "multi_step_kb_format",
                    },
                )
                trace.append(WorkflowStep.VERIFY)
                ledger.append(
                    agent="critic",
                    workflow_step=WorkflowStep.VERIFY.value,
                    payload={
                        "passed": True,
                        "confidence": final_confidence,
                        "issues": [],
                        "revised_answer": final_answer,
                        "mode": "multi_step_kb_format",
                    },
                )
            else:
                summary = self.summarizer.run(
                    SummarizeRequest(
                        run_id=user.run_id,
                        question=user.question,
                        plan_steps=plan_res.steps,
                        step_answers=step_answers,
                    )
                )
                ledger.append(
                    agent="summarizer",
                    workflow_step=WorkflowStep.FINALIZE.value,
                    payload={
                        "draft_answer": summary.answer,
                        "confidence": summary.confidence,
                    },
                )

                trace.append(WorkflowStep.VERIFY)
                verify = self.critic.run(
                    VerifyRequest(
                        run_id=user.run_id,
                        question=user.question,
                        draft_answer=summary.answer,
                        step_answers=step_answers,
                        chunk_ids=list(dict.fromkeys(chunk_ids)),
                    )
                )
                ledger.append(
                    agent="critic",
                    workflow_step=WorkflowStep.VERIFY.value,
                    payload={
                        "passed": verify.passed,
                        "confidence": verify.confidence,
                        "issues": verify.issues,
                        "revised_answer": verify.revised_answer,
                    },
                )

                final_answer = verify.revised_answer or summary.answer
                final_confidence = (
                    verify.confidence
                    if verify.passed
                    else min(summary.confidence, verify.confidence)
                )
                verify_passed = verify.passed
                verify_issues = list(verify.issues)

                final_answer, extra_issues = reconcile_final_answer(
                    final_answer, step_answers
                )
                if extra_issues:
                    verify_issues.extend(extra_issues)
                    verify_passed = True
        else:
            summary = self.summarizer.run(
                SummarizeRequest(
                    run_id=user.run_id,
                    question=user.question,
                    plan_steps=plan_res.steps,
                    step_answers=step_answers,
                )
            )
            ledger.append(
                agent="summarizer",
                workflow_step=WorkflowStep.FINALIZE.value,
                payload={
                    "draft_answer": summary.answer,
                    "confidence": summary.confidence,
                },
            )

            trace.append(WorkflowStep.VERIFY)
            verify = self.critic.run(
                VerifyRequest(
                    run_id=user.run_id,
                    question=user.question,
                    draft_answer=summary.answer,
                    step_answers=step_answers,
                    chunk_ids=list(dict.fromkeys(chunk_ids)),
                )
            )
            ledger.append(
                agent="critic",
                workflow_step=WorkflowStep.VERIFY.value,
                payload={
                    "passed": verify.passed,
                    "confidence": verify.confidence,
                    "issues": verify.issues,
                    "revised_answer": verify.revised_answer,
                },
            )

            final_answer = verify.revised_answer or summary.answer
            final_confidence = (
                verify.confidence
                if verify.passed
                else min(summary.confidence, verify.confidence)
            )
            verify_passed = verify.passed
            verify_issues = list(verify.issues)

            final_answer, extra_issues = reconcile_final_answer(
                final_answer, step_answers
            )
            if extra_issues:
                verify_issues.extend(extra_issues)
                verify_passed = True

        return FinalAnswerPackage(
            run_id=user.run_id,
            question=user.question,
            answer=final_answer,
            confidence=final_confidence,
            plan_steps=plan_res.steps,
            step_answers=step_answers,
            workflow_trace=trace,
            chunk_ids_used=list(dict.fromkeys(chunk_ids)),
            verify_passed=verify_passed,
            verify_issues=verify_issues,
            evidence_ledger_path=str(ledger.path),
            route_decision=route_res.decision,
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
    if package.route_decision is not None:
        lines.append(f"Route: {package.route_decision.value}")
    if package.verify_passed is not None:
        lines.append(f"Verify passed: {'Yes' if package.verify_passed else 'No'}")
    if package.verify_issues:
        lines.append(f"Verify issues: {', '.join(package.verify_issues)}")
    lines.append(f"\nRun id: {package.run_id}")
    if package.evidence_ledger_path:
        lines.append(f"Evidence ledger: {package.evidence_ledger_path}")
    lines.append(f"Trace: {', '.join(s.value for s in package.workflow_trace)}")
    return "\n".join(lines)
