"""Summarizer agent — final answer from step outputs."""

from __future__ import annotations

from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from src.contracts.messages import SummarizeRequest, SummarizeResponse
from langchain_core.output_parsers import StrOutputParser

from src.llm import create_chat_llm
from src.prompt_template import (
    summary_human_message,
    summary_input_variables,
    summary_slm_output_format_addon,
    summary_system_message,
)
from src.slm_helpers import is_ollama_provider, parse_summary_response
from src.utils import PlanSummaryFormat

AGENT_ID = "summarizer"
ROLE = "Summarizer — combine step answers into a final response"


class SummarizerAgent:
    def __init__(self, *, agent_id: str = AGENT_ID, role: str = ROLE):
        self.agent_id = agent_id
        self.role = role

    def run(self, request: SummarizeRequest) -> SummarizeResponse:
        plan = f"[{', '.join(request.plan_steps)}]"
        memory = ""
        for step in request.step_answers:
            memory += (
                f"Task: {step.plan_step}\n"
                f"Question: {step.task}\n"
                f"Answer: {step.answer}\n"
                f"Confident score: {step.confidence}\n\n"
            )

        system_message = summary_system_message
        if is_ollama_provider():
            system_message += summary_slm_output_format_addon
        messages = [
            SystemMessagePromptTemplate.from_template(system_message),
            HumanMessagePromptTemplate.from_template(summary_human_message),
        ]
        prompt = ChatPromptTemplate(
            input_variables=summary_input_variables,
            messages=messages,
        )
        llm = create_chat_llm(temperature=0.0)
        inputs = {"question": request.question, "plan": plan, "memory": memory}
        if is_ollama_provider():
            text = (prompt | llm | StrOutputParser()).invoke(inputs)
            parsed = parse_summary_response(text)
            answer = parsed["answer"]
            confidence = int(parsed["score"] or 0)
            summary = parsed["output"] or text.strip()
        else:
            out = (prompt | llm.with_structured_output(PlanSummaryFormat)).invoke(inputs)
            answer = out.answer
            confidence = int(out.score or 0)
            summary = out.output

        return SummarizeResponse(
            run_id=request.run_id,
            answer=answer,
            confidence=confidence,
            summary=summary,
        )
