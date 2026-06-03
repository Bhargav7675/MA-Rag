"""RAG step agent — retrieve, extract, and QA for one sub-question."""

from __future__ import annotations

from typing import Any

from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from agents.rag import build_rag_agent
from src.contracts.messages import StepAnswer, StepTaskType
from src.llm import create_chat_llm
from src.prompt_template import (
    aggregate_human_message,
    aggregate_input_variables,
    aggregate_system_message,
)
from src.utils import QAAnswerFormat

AGENT_ID = "rag_step"
ROLE = "RAG — grounded step answer from retrieved evidence"


class RagStepAgent:
    def __init__(
        self,
        retriever_tool: Any,
        *,
        agent_id: str = AGENT_ID,
        role: str = ROLE,
    ):
        self._rag_graph = build_rag_agent(retriever_tool=retriever_tool)
        self.agent_id = agent_id
        self.role = role

    def run(
        self,
        *,
        run_id: str,
        step_index: int,
        plan_step: str,
        task: str,
        task_type: StepTaskType,
    ) -> StepAnswer:
        if task_type == StepTaskType.AGGREGATE:
            return self._run_aggregate(
                run_id=run_id,
                step_index=step_index,
                plan_step=plan_step,
                task=task,
            )
        return self._run_rag(
            run_id=run_id,
            step_index=step_index,
            plan_step=plan_step,
            task=task,
        )

    def _run_rag(
        self,
        *,
        run_id: str,
        step_index: int,
        plan_step: str,
        task: str,
    ) -> StepAnswer:
        out = self._rag_graph.invoke({"question": task})
        raw = out["final_raw_answer"]
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump()
        elif not isinstance(raw, dict):
            raw = dict(raw)

        doc_ids = [str(d) for d in (out.get("doc_ids") or [])]
        return StepAnswer(
            step_index=step_index,
            plan_step=plan_step,
            task=task,
            analysis=raw.get("analysis", ""),
            answer=raw.get("answer", ""),
            success=str(raw.get("success", "No")).lower() == "yes",
            confidence=int(raw.get("rating") or 0),
            doc_ids=doc_ids,
        )

    def _run_aggregate(
        self,
        *,
        run_id: str,
        step_index: int,
        plan_step: str,
        task: str,
    ) -> StepAnswer:
        messages = [
            SystemMessagePromptTemplate.from_template(aggregate_system_message),
            HumanMessagePromptTemplate.from_template(aggregate_human_message),
        ]
        prompt = ChatPromptTemplate(
            input_variables=aggregate_input_variables,
            messages=messages,
        )
        llm = create_chat_llm(temperature=0.0)
        chain = prompt | llm.with_structured_output(QAAnswerFormat)
        raw = chain.invoke({"question": task})
        return StepAnswer(
            step_index=step_index,
            plan_step=plan_step,
            task=task,
            analysis=raw.analysis,
            answer=raw.answer,
            success=str(raw.success).lower() == "yes",
            confidence=int(raw.rating or 0),
            doc_ids=[],
        )
