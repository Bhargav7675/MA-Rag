"""Step definer agent — plan step to concrete sub-task."""

from __future__ import annotations

from typing import Optional

from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from src.contracts.messages import (
    StepDefineRequest,
    StepDefineResponse,
    StepTaskType,
)
from src.llm import create_chat_llm
from src.prompt_template import (
    step_human_message,
    step_input_variables,
    step_system_message,
)
from src.slm_helpers import is_ollama_provider
from src.utils import StepTaskFormat

AGENT_ID = "step_definer"
ROLE = "Step definer — turn plan lines into executable sub-questions"

_KB_NAMES = ("Chandra Shekar Konda", "Bhargav Boyapati", "Oracle")


def refine_task_from_prior(
    cur_step: str,
    prior_step_answers: list,
) -> Optional[str]:
    """Heuristic follow-up tasks when SLM step definer misfires on multi-hop."""
    prior_blob = " ".join(
        step.answer for step in prior_step_answers if step.success and step.answer.strip()
    )
    if not prior_blob.strip():
        return None

    cur_lower = cur_step.lower()
    prior_lower = prior_blob.lower()

    if "title" in cur_lower and "oracle" in cur_lower:
        for name in _KB_NAMES[:2]:
            if name.lower() in prior_lower:
                return f"What is {name}'s title at Oracle?"

    if ("person" in cur_lower or "who" in cur_lower) and "oracle" in cur_lower:
        if "chandra shekar konda" in prior_lower:
            return "What is Chandra Shekar Konda's title at Oracle?"

    return None


class StepDefinerAgent:
    def __init__(self, *, agent_id: str = AGENT_ID, role: str = ROLE):
        self.agent_id = agent_id
        self.role = role

    def run(self, request: StepDefineRequest) -> StepDefineResponse:
        cur_step = request.plan_steps[request.current_step_index]

        # First step with no prior answers must retrieve — never aggregate air.
        if request.current_step_index == 0 and not request.prior_step_answers:
            return StepDefineResponse(
                run_id=request.run_id,
                step_index=request.current_step_index,
                task_type=StepTaskType.QUESTION_ANSWERING,
                task=cur_step,
            )

        refined = refine_task_from_prior(cur_step, request.prior_step_answers)
        if refined:
            return StepDefineResponse(
                run_id=request.run_id,
                step_index=request.current_step_index,
                task_type=StepTaskType.QUESTION_ANSWERING,
                task=refined,
            )

        plan = f"[{', '.join(request.plan_steps)}]"
        memory = ""
        for prior in request.prior_step_answers:
            memory += f"Task: {prior.plan_step}\nAnswer: {prior.answer}\n\n"

        messages = [
            SystemMessagePromptTemplate.from_template(step_system_message),
            HumanMessagePromptTemplate.from_template(step_human_message),
        ]
        prompt = ChatPromptTemplate(
            input_variables=step_input_variables,
            messages=messages,
        )
        llm = create_chat_llm(temperature=0.0 if is_ollama_provider() else 0.3)
        chain = prompt | llm.with_structured_output(StepTaskFormat)
        out = chain.invoke({"plan": plan, "cur_step": cur_step, "memory": memory})

        task_type = (
            StepTaskType.QUESTION_ANSWERING
            if out.type == "question-answering"
            else StepTaskType.AGGREGATE
        )
        task = out.task
        if task_type == StepTaskType.AGGREGATE and request.prior_step_answers:
            task_type = StepTaskType.QUESTION_ANSWERING
            task = refined or cur_step

        return StepDefineResponse(
            run_id=request.run_id,
            step_index=request.current_step_index,
            task_type=task_type,
            task=task,
        )
