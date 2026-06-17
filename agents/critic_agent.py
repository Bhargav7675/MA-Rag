"""Critic / Verifier — faithfulness check before finalize."""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from src.contracts.messages import StepAnswer, VerifyRequest, VerifyResponse
from src.llm import create_chat_llm, is_agent_ollama
from src.prompt_template import (
    critic_human_message,
    critic_input_variables,
    critic_multi_step_addon,
    critic_slm_output_format_addon,
    critic_system_message,
)
from src.workflow.finalize_helpers import reconcile_final_answer
from src.slm_helpers import parse_verify_response
from src.utils import VerifyFormat

AGENT_ID = "critic"
ROLE = "Critic / Verifier — check draft answer against step evidence"


class CriticAgent:
    def __init__(self, *, agent_id: str = AGENT_ID, role: str = ROLE):
        self.agent_id = agent_id
        self.role = role

    def run(self, request: VerifyRequest) -> VerifyResponse:
        if not request.draft_answer.strip():
            return VerifyResponse(
                run_id=request.run_id,
                passed=False,
                confidence=0,
                issues=["Empty draft answer"],
            )

        if not request.step_answers:
            return VerifyResponse(
                run_id=request.run_id,
                passed=True,
                confidence=5,
                issues=[],
                revised_answer=request.draft_answer,
            )

        step_evidence = self._format_step_evidence(request.step_answers)
        chunk_ids = ", ".join(request.chunk_ids) or "none"
        multi_step = len(request.step_answers) > 1

        system_message = critic_system_message
        if multi_step:
            system_message += critic_multi_step_addon
        if is_agent_ollama("critic"):
            system_message += critic_slm_output_format_addon
        messages = [
            SystemMessagePromptTemplate.from_template(system_message),
            HumanMessagePromptTemplate.from_template(critic_human_message),
        ]
        prompt = ChatPromptTemplate(
            input_variables=critic_input_variables,
            messages=messages,
        )
        llm = create_chat_llm(agent_id="critic", temperature=0.0)
        inputs = {
            "question": request.question,
            "draft_answer": request.draft_answer,
            "step_evidence": step_evidence,
            "chunk_ids": chunk_ids,
        }

        if is_agent_ollama("critic"):
            text = (prompt | llm | StrOutputParser()).invoke(inputs)
            parsed = parse_verify_response(text)
            passed = parsed["passed"]
            confidence = int(parsed["confidence"] or 0)
            issues = parsed["issues"]
            revised = parsed["revised_answer"] or request.draft_answer
        else:
            raw = (prompt | llm.with_structured_output(VerifyFormat)).invoke(inputs)
            passed = str(raw.passed).lower() == "yes"
            confidence = int(raw.confidence or 0)
            issues = raw.issues
            revised = raw.revised_answer or request.draft_answer

        revised, reconcile_issues = reconcile_final_answer(
            revised, request.step_answers
        )
        if reconcile_issues:
            issues = list(issues) + reconcile_issues
            passed = True

        return VerifyResponse(
            run_id=request.run_id,
            passed=passed,
            confidence=confidence,
            issues=issues,
            revised_answer=revised,
        )

    @staticmethod
    def _format_step_evidence(step_answers: list[StepAnswer]) -> str:
        parts = []
        for step in step_answers:
            parts.append(
                f"Step {step.step_index + 1}: {step.plan_step}\n"
                f"Task: {step.task}\n"
                f"Answer: {step.answer}\n"
                f"Success: {'Yes' if step.success else 'No'}\n"
                f"Confidence: {step.confidence}\n"
                f"Doc ids: {', '.join(step.doc_ids) or 'none'}"
            )
        return "\n\n".join(parts)
