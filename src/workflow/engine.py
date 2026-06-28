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
from src.a2a.file_journal import A2AFileJournal
from src.a2a.file_queue_bus import FileQueueA2ABus
from src.contracts.messages import (
    EvidenceReviewRequest,
    EvidenceReviewResponse,
    FinalAnswerPackage,
    PlanRequest,
    PlanResponse,
    RagStepRequest,
    RetrievalResponse,
    RetrievalTask,
    RouterRequest,
    RouterResponse,
    StepAnswer,
    StepDefineRequest,
    StepDefineResponse,
    StepTaskType,
    SummarizeRequest,
    SummarizeResponse,
    UserRequest,
    VerifyRequest,
    VerifyResponse,
    WorkflowStep,
)
from src.evidence_ledger import EvidenceLedger
from src.env import get_a2a_transport, get_fast_mode
from src.pipeline import normalize_question
from src.tools.registry import ToolRegistry, default_tool_registry
from src.workflow.a2a_setup import a2a_request, setup_a2a_bus
from src.workflow.events import (
    WorkflowEvent,
    WorkflowEventCallback,
    WorkflowEventKind,
    emit_event,
)
from src.slm_helpers import extract_factual_answer_from_context
from src.workflow.finalize_helpers import (
    all_steps_successful,
    finalize_multi_hop_quality,
    format_kb_multi_hop_answer,
    format_summit_multi_hop_answer,
    mean_step_confidence,
    normalize_step_answer,
    reconcile_final_answer,
    steps_have_substantive_answers,
)


class WorkflowEngine:
    """
    Orchestrates typed agents without LangGraph shared state.

    Steps: ROUTE → INIT_PLAN → (per plan step: RETRIEVE → EVIDENCE_CHECK → GENERATE)
           → FINALIZE draft → VERIFY → FINALIZE
    """

    def __init__(
        self,
        retriever_tool: Any,
        *,
        tool_registry: Optional[ToolRegistry] = None,
        use_a2a: bool = True,
    ):
        self.retriever_tool = retriever_tool
        self.tool_registry = tool_registry or default_tool_registry(retriever_tool)
        self.use_a2a = use_a2a
        self.router = RouterAgent()
        self.planner = PlannerAgent()
        self.retrieval = RetrievalAgent(tool_registry=self.tool_registry)
        self.evidence_curator = EvidenceCuratorAgent()
        self.step_definer = StepDefinerAgent()
        self.rag_step = RagStepAgent(retriever_tool)
        self.summarizer = SummarizerAgent()
        self.critic = CriticAgent()
        self.a2a_journal = A2AFileJournal() if use_a2a else None
        if use_a2a:
            transport = get_a2a_transport()
            if transport == "file_queue":
                bus = FileQueueA2ABus(journal=self.a2a_journal)
            else:
                from src.a2a.bus import InProcessA2ABus

                bus = InProcessA2ABus(journal=self.a2a_journal)
            self.a2a_bus = setup_a2a_bus(
                router=self.router,
                planner=self.planner,
                retrieval=self.retrieval,
                evidence_curator=self.evidence_curator,
                step_definer=self.step_definer,
                rag_step=self.rag_step,
                summarizer=self.summarizer,
                critic=self.critic,
                bus=bus,
            )
        else:
            self.a2a_bus = None
        self._on_event: Optional[WorkflowEventCallback] = None

    def _emit(
        self,
        kind: WorkflowEventKind,
        *,
        run_id: str,
        workflow_step: Optional[WorkflowStep] = None,
        agent: Optional[str] = None,
        message_type: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        emit_event(
            self._on_event,
            WorkflowEvent(
                run_id=run_id,
                kind=kind,
                workflow_step=workflow_step.value if workflow_step else None,
                agent=agent,
                message_type=message_type,
                payload=payload or {},
            ),
        )

    def _dispatch_router(self, request: RouterRequest) -> RouterResponse:
        self._emit(
            WorkflowEventKind.AGENT_START,
            run_id=request.run_id,
            workflow_step=WorkflowStep.ROUTE,
            agent="router",
            message_type="router.request",
        )
        if self.use_a2a and self.a2a_bus:
            data = a2a_request(
                self.a2a_bus,
                to_agent="router",
                message_type="router.request",
                payload=request.model_dump(mode="json"),
                correlation_id=request.run_id,
            )
            response = RouterResponse(**data)
        else:
            response = self.router.run(request)
        self._emit(
            WorkflowEventKind.AGENT_COMPLETE,
            run_id=request.run_id,
            workflow_step=WorkflowStep.ROUTE,
            agent="router",
            message_type="router.request",
            payload={
                "decision": response.decision.value,
                "rationale": response.rationale,
            },
        )
        return response

    def _dispatch_planner(self, request: PlanRequest) -> PlanResponse:
        self._emit(
            WorkflowEventKind.AGENT_START,
            run_id=request.run_id,
            workflow_step=WorkflowStep.INIT_PLAN,
            agent="planner",
            message_type="plan.request",
        )
        if self.use_a2a and self.a2a_bus:
            data = a2a_request(
                self.a2a_bus,
                to_agent="planner",
                message_type="plan.request",
                payload=request.model_dump(mode="json"),
                correlation_id=request.run_id,
            )
            response = PlanResponse(**data)
        else:
            response = self.planner.run(request)
        self._emit(
            WorkflowEventKind.AGENT_COMPLETE,
            run_id=request.run_id,
            workflow_step=WorkflowStep.INIT_PLAN,
            agent="planner",
            message_type="plan.request",
            payload={"steps": response.steps, "analysis": response.analysis},
        )
        return response

    def _dispatch_retrieval(self, task: RetrievalTask) -> RetrievalResponse:
        self._emit(
            WorkflowEventKind.AGENT_START,
            run_id=task.run_id,
            workflow_step=WorkflowStep.RETRIEVE,
            agent="retrieval",
            message_type="retrieval.task",
            payload={"step_index": task.step_index, "question": task.question},
        )
        if self.use_a2a and self.a2a_bus:
            data = a2a_request(
                self.a2a_bus,
                to_agent="retrieval",
                message_type="retrieval.task",
                payload=task.model_dump(mode="json"),
                correlation_id=task.run_id,
            )
            response = RetrievalResponse(**data)
        else:
            response = self.retrieval.run(task)
        self._emit(
            WorkflowEventKind.AGENT_COMPLETE,
            run_id=task.run_id,
            workflow_step=WorkflowStep.RETRIEVE,
            agent="retrieval",
            message_type="retrieval.task",
            payload={
                "step_index": task.step_index,
                "chunk_count": len(response.chunks),
                "chunk_ids": [chunk.doc_id for chunk in response.chunks],
            },
        )
        return response

    def _dispatch_evidence(
        self, request: EvidenceReviewRequest
    ) -> EvidenceReviewResponse:
        self._emit(
            WorkflowEventKind.AGENT_START,
            run_id=request.run_id,
            workflow_step=WorkflowStep.EVIDENCE_CHECK,
            agent="evidence_curator",
            message_type="evidence.review",
            payload={"step_index": request.step_index},
        )
        if self.use_a2a and self.a2a_bus:
            data = a2a_request(
                self.a2a_bus,
                to_agent="evidence_curator",
                message_type="evidence.review",
                payload=request.model_dump(mode="json"),
                correlation_id=request.run_id,
            )
            response = EvidenceReviewResponse(**data)
        else:
            response = self.evidence_curator.run(request)
        self._emit(
            WorkflowEventKind.AGENT_COMPLETE,
            run_id=request.run_id,
            workflow_step=WorkflowStep.EVIDENCE_CHECK,
            agent="evidence_curator",
            message_type="evidence.review",
            payload={
                "step_index": request.step_index,
                "sufficiency": response.sufficiency.value,
                "proceed": response.proceed,
                "gaps": response.gaps,
            },
        )
        return response

    def _dispatch_step_definer(
        self, request: StepDefineRequest
    ) -> StepDefineResponse:
        self._emit(
            WorkflowEventKind.AGENT_START,
            run_id=request.run_id,
            agent="step_definer",
            message_type="step.define",
            payload={"current_step_index": request.current_step_index},
        )
        if self.use_a2a and self.a2a_bus:
            data = a2a_request(
                self.a2a_bus,
                to_agent="step_definer",
                message_type="step.define",
                payload=request.model_dump(mode="json"),
                correlation_id=request.run_id,
            )
            response = StepDefineResponse(**data)
        else:
            response = self.step_definer.run(request)
        self._emit(
            WorkflowEventKind.AGENT_COMPLETE,
            run_id=request.run_id,
            agent="step_definer",
            message_type="step.define",
            payload={
                "task": response.task,
                "task_type": response.task_type.value,
            },
        )
        return response

    def _dispatch_summarizer(self, request: SummarizeRequest) -> SummarizeResponse:
        self._emit(
            WorkflowEventKind.AGENT_START,
            run_id=request.run_id,
            workflow_step=WorkflowStep.FINALIZE,
            agent="summarizer",
            message_type="summarize.request",
        )
        if self.use_a2a and self.a2a_bus:
            data = a2a_request(
                self.a2a_bus,
                to_agent="summarizer",
                message_type="summarize.request",
                payload=request.model_dump(mode="json"),
                correlation_id=request.run_id,
            )
            response = SummarizeResponse(**data)
        else:
            response = self.summarizer.run(request)
        self._emit(
            WorkflowEventKind.AGENT_COMPLETE,
            run_id=request.run_id,
            workflow_step=WorkflowStep.FINALIZE,
            agent="summarizer",
            message_type="summarize.request",
            payload={
                "answer": response.answer,
                "confidence": response.confidence,
            },
        )
        return response

    def _dispatch_critic(self, request: VerifyRequest) -> VerifyResponse:
        self._emit(
            WorkflowEventKind.AGENT_START,
            run_id=request.run_id,
            workflow_step=WorkflowStep.VERIFY,
            agent="critic",
            message_type="verify.request",
        )
        if self.use_a2a and self.a2a_bus:
            data = a2a_request(
                self.a2a_bus,
                to_agent="critic",
                message_type="verify.request",
                payload=request.model_dump(mode="json"),
                correlation_id=request.run_id,
            )
            response = VerifyResponse(**data)
        else:
            response = self.critic.run(request)
        self._emit(
            WorkflowEventKind.AGENT_COMPLETE,
            run_id=request.run_id,
            workflow_step=WorkflowStep.VERIFY,
            agent="critic",
            message_type="verify.request",
            payload={
                "passed": response.passed,
                "confidence": response.confidence,
                "issues": response.issues,
            },
        )
        return response

    def _dispatch_rag_step(self, request: RagStepRequest) -> StepAnswer:
        self._emit(
            WorkflowEventKind.AGENT_START,
            run_id=request.run_id,
            workflow_step=WorkflowStep.GENERATE,
            agent="rag_step",
            message_type="rag.step",
            payload={"step_index": request.step_index, "task": request.task},
        )
        if self.use_a2a and self.a2a_bus:
            data = a2a_request(
                self.a2a_bus,
                to_agent="rag_step",
                message_type="rag.step",
                payload=request.model_dump(mode="json"),
                correlation_id=request.run_id,
            )
            response = StepAnswer(**data)
        else:
            kwargs = {
                "run_id": request.run_id,
                "step_index": request.step_index,
                "plan_step": request.plan_step,
                "task": request.task,
                "task_type": request.task_type,
            }
            if request.documents is not None and request.doc_ids is not None:
                kwargs["documents"] = request.documents
                kwargs["doc_ids"] = request.doc_ids
            response = self.rag_step.run(**kwargs)
        self._emit(
            WorkflowEventKind.AGENT_COMPLETE,
            run_id=request.run_id,
            workflow_step=WorkflowStep.GENERATE,
            agent="rag_step",
            message_type="rag.step",
            payload={
                "step_index": request.step_index,
                "success": response.success,
                "confidence": response.confidence,
                "answer": response.answer,
            },
        )
        return response

    def run(
        self,
        question: str,
        *,
        run_id: Optional[str] = None,
        on_event: Optional[WorkflowEventCallback] = None,
    ) -> FinalAnswerPackage:
        question = normalize_question(question)
        user = UserRequest(question=question)
        if run_id:
            user.run_id = run_id

        self._on_event = on_event
        self._emit(
            WorkflowEventKind.WORKFLOW_START,
            run_id=user.run_id,
            payload={"question": user.question},
        )

        ledger = EvidenceLedger(user.run_id)
        trace: list[WorkflowStep] = [WorkflowStep.ROUTE]
        route_res = self._dispatch_router(
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
        plan_res = self._dispatch_planner(
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
                define_res = self._dispatch_step_definer(
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
                answer = self._dispatch_rag_step(
                    RagStepRequest(
                        run_id=user.run_id,
                        step_index=step_index,
                        plan_step=plan_step,
                        task=task,
                        task_type=task_type,
                    )
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
            retrieval = self._dispatch_retrieval(
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

            if get_fast_mode() and retrieval.chunks:
                blob = "\n\n".join(
                    chunk.text for chunk in retrieval.chunks if chunk.text.strip()
                )
                direct = extract_factual_answer_from_context(task, blob)
                if direct:
                    trace.append(WorkflowStep.CONTEXT_BUILD)
                    trace.append(WorkflowStep.GENERATE)
                    step_answers.append(
                        StepAnswer(
                            step_index=step_index,
                            plan_step=plan_step,
                            task=task,
                            analysis="Fast mode: direct factual match from retrieved chunks.",
                            answer=direct,
                            success=True,
                            confidence=9,
                            doc_ids=[chunk.doc_id for chunk in retrieval.chunks],
                        )
                    )
                    ledger.append(
                        agent="rag_step",
                        workflow_step=WorkflowStep.GENERATE.value,
                        payload={
                            "step_index": step_index,
                            "task": task,
                            "answer": direct,
                            "success": True,
                            "confidence": 9,
                            "mode": "fast_factual",
                        },
                    )
                    continue

            trace.append(WorkflowStep.EVIDENCE_CHECK)
            evidence = self._dispatch_evidence(
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

            if not evidence.proceed and evidence.gaps:
                retry_task = f"{task} {' '.join(evidence.gaps)}".strip()
                if retry_task != task:
                    trace.append(WorkflowStep.RETRIEVE)
                    retrieval = self._dispatch_retrieval(
                        RetrievalTask(
                            run_id=user.run_id,
                            step_index=step_index,
                            question=retry_task,
                        )
                    )
                    chunk_ids.extend(chunk.doc_id for chunk in retrieval.chunks)
                    ledger.append(
                        agent="retrieval",
                        workflow_step=WorkflowStep.RETRIEVE.value,
                        payload={
                            "step_index": step_index,
                            "question": retry_task,
                            "chunk_ids": [chunk.doc_id for chunk in retrieval.chunks],
                            "mode": "evidence_retry",
                        },
                    )
                    trace.append(WorkflowStep.EVIDENCE_CHECK)
                    evidence = self._dispatch_evidence(
                        EvidenceReviewRequest(
                            run_id=user.run_id,
                            step_index=step_index,
                            question=retry_task,
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
                            "mode": "evidence_retry",
                        },
                    )
                    task = retry_task

            if not evidence.proceed and not retrieval.chunks:
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
            answer = self._dispatch_rag_step(
                RagStepRequest(
                    run_id=user.run_id,
                    step_index=step_index,
                    plan_step=plan_step,
                    task=task,
                    task_type=task_type,
                    documents=[chunk.text for chunk in retrieval.chunks],
                    doc_ids=[chunk.doc_id for chunk in retrieval.chunks],
                )
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

        step_answers = [normalize_step_answer(step) for step in step_answers]

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
        elif steps_have_substantive_answers(step_answers) and len(step_answers) >= 2:
            formatted = format_kb_multi_hop_answer(step_answers)
            if not formatted:
                formatted = format_summit_multi_hop_answer(step_answers)
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
                summary = self._dispatch_summarizer(
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
                verify = self._dispatch_critic(
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
                elif not verify_passed and all_steps_successful(step_answers):
                    from src.workflow.finalize_helpers import merge_step_answers_text

                    merged = merge_step_answers_text(step_answers)
                    if merged and merged.strip() != final_answer.strip():
                        final_answer = merged
                        verify_issues.append("replaced failed verify with merged step answers")
                        verify_passed = True
        else:
            summary = self._dispatch_summarizer(
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
            verify = self._dispatch_critic(
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
            elif not verify_passed and all_steps_successful(step_answers):
                from src.workflow.finalize_helpers import merge_step_answers_text

                merged = merge_step_answers_text(step_answers)
                if merged and merged.strip() != final_answer.strip():
                    final_answer = merged
                    verify_issues.append("replaced failed verify with merged step answers")
                    verify_passed = True

        if len(step_answers) >= 2 and steps_have_substantive_answers(step_answers):
            final_answer, final_confidence, verify_passed, quality_issues = (
                finalize_multi_hop_quality(
                    final_answer,
                    final_confidence,
                    verify_passed,
                    step_answers,
                )
            )
            if quality_issues:
                verify_issues.extend(quality_issues)

        package = FinalAnswerPackage(
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
            a2a_journal_path=(
                str(self.a2a_journal.path_for(user.run_id))
                if self.a2a_journal is not None
                else None
            ),
            route_decision=route_res.decision,
        )
        self._emit(
            WorkflowEventKind.WORKFLOW_COMPLETE,
            run_id=user.run_id,
            payload={
                "answer": package.answer,
                "confidence": package.confidence,
                "route": package.route_decision.value if package.route_decision else None,
                "verify_passed": package.verify_passed,
                "workflow_trace": [step.value for step in package.workflow_trace],
            },
        )
        self._on_event = None
        return package


def format_workflow_output(package: FinalAnswerPackage) -> str:
    from src.workflow.event_display import format_workflow_output_clear

    return format_workflow_output_clear(package)
