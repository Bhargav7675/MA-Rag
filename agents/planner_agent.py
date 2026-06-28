"""Planner agent with typed contracts (wraps existing plan_agent logic)."""

from __future__ import annotations

from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from src.contracts.messages import PlanRequest, PlanResponse, RouteDecision
from src.llm import create_chat_llm, is_agent_ollama
from src.prompt_template import (
    planing_human_message,
    planing_input_variables,
    planing_project_kb_addon,
    planing_system_message,
)
from src.slm_helpers import canonical_kb_plan, heuristic_multi_hop_plan, simplify_plan_for_slm
from src.utils import PlanFormat

AGENT_ID = "planner"
ROLE = "Supervisor / Planner — decompose questions into ordered reasoning steps"


class PlannerAgent:
    def __init__(self, *, agent_id: str = AGENT_ID, role: str = ROLE):
        self.agent_id = agent_id
        self.role = role

    def run(self, request: PlanRequest) -> PlanResponse:
        canonical = canonical_kb_plan(request.question)
        if canonical:
            return PlanResponse(
                run_id=request.run_id,
                analysis="Canonical MA-RAG knowledge-base multi-hop plan.",
                steps=canonical,
            )

        heuristic = heuristic_multi_hop_plan(request.question)
        if request.route_decision == RouteDecision.MULTI_HOP_RAG and heuristic:
            return PlanResponse(
                run_id=request.run_id,
                analysis="Heuristic multi-hop plan from question structure.",
                steps=heuristic,
            )

        if request.route_decision == RouteDecision.SIMPLE_RAG:
            q = request.question.strip()
            if not q.endswith("?"):
                q = f"{q}?"
            return PlanResponse(
                run_id=request.run_id,
                analysis="Single-step factual lookup.",
                steps=[q],
            )

        memory = "empty"
        if request.past_trial_summaries:
            parts = [
                f"Trial {i}:\n{summary}"
                for i, summary in enumerate(request.past_trial_summaries)
            ]
            memory = "\n".join(parts)

        system_message = planing_system_message + planing_project_kb_addon
        messages = [
            SystemMessagePromptTemplate.from_template(system_message),
            HumanMessagePromptTemplate.from_template(planing_human_message),
        ]
        prompt = ChatPromptTemplate(
            input_variables=planing_input_variables,
            messages=messages,
        )
        temperature = 0.0 if is_agent_ollama("planner") else 0.3
        llm = create_chat_llm(agent_id="planner", temperature=temperature)
        chain = prompt | llm.with_structured_output(PlanFormat)
        output = chain.invoke({"question": request.question, "memory": memory})

        if request.route_decision == RouteDecision.MULTI_HOP_RAG:
            steps = output.step
            if is_agent_ollama("planner") and len(steps) > 3:
                q = request.question.strip()
                if not q.endswith("?"):
                    q = f"{q}?"
                steps = [q]
        else:
            steps = simplify_plan_for_slm(request.question, output.step)

        return PlanResponse(
            run_id=request.run_id,
            analysis=output.analysis,
            steps=steps,
        )
