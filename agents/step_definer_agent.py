"""Step definer agent — plan step to concrete sub-task."""

from __future__ import annotations

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
from src.utils import StepTaskFormat

AGENT_ID = "step_definer"
ROLE = "Step definer — turn plan lines into executable sub-questions"


class StepDefinerAgent:
    def __init__(self, *, agent_id: str = AGENT_ID, role: str = ROLE):
        self.agent_id = agent_id
        self.role = role

    def run(self, request: StepDefineRequest) -> StepDefineResponse:
        plan = f"[{', '.join(request.plan_steps)}]"
        memory = ""
        for prior in request.prior_step_answers:
            memory += f"Task: {prior.plan_step}\nAnswer: {prior.answer}\n\n"

        cur_step = request.plan_steps[request.current_step_index]
        messages = [
            SystemMessagePromptTemplate.from_template(step_system_message),
            HumanMessagePromptTemplate.from_template(step_human_message),
        ]
        prompt = ChatPromptTemplate(
            input_variables=step_input_variables,
            messages=messages,
        )
        llm = create_chat_llm(temperature=0.3)
        chain = prompt | llm.with_structured_output(StepTaskFormat)
        out = chain.invoke({"plan": plan, "cur_step": cur_step, "memory": memory})

        task_type = (
            StepTaskType.QUESTION_ANSWERING
            if out.type == "question-answering"
            else StepTaskType.AGGREGATE
        )

        return StepDefineResponse(
            run_id=request.run_id,
            step_index=request.current_step_index,
            task_type=task_type,
            task=out.task,
        )
