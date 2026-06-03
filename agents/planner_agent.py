"""Planner agent with typed contracts (wraps existing plan_agent logic)."""

from __future__ import annotations

from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from src.contracts.messages import PlanRequest, PlanResponse
from src.llm import create_chat_llm
from src.prompt_template import (
    planing_human_message,
    planing_input_variables,
    planing_system_message,
)
from src.utils import PlanFormat

AGENT_ID = "planner"
ROLE = "Supervisor / Planner — decompose questions into ordered reasoning steps"


class PlannerAgent:
    def __init__(self, *, agent_id: str = AGENT_ID, role: str = ROLE):
        self.agent_id = agent_id
        self.role = role

    def run(self, request: PlanRequest) -> PlanResponse:
        memory = "empty"
        if request.past_trial_summaries:
            parts = [
                f"Trial {i}:\n{summary}"
                for i, summary in enumerate(request.past_trial_summaries)
            ]
            memory = "\n".join(parts)

        messages = [
            SystemMessagePromptTemplate.from_template(planing_system_message),
            HumanMessagePromptTemplate.from_template(planing_human_message),
        ]
        prompt = ChatPromptTemplate(
            input_variables=planing_input_variables,
            messages=messages,
        )
        llm = create_chat_llm(temperature=0.3)
        chain = prompt | llm.with_structured_output(PlanFormat)
        output = chain.invoke({"question": request.question, "memory": memory})

        return PlanResponse(
            run_id=request.run_id,
            analysis=output.analysis,
            steps=output.step,
        )
