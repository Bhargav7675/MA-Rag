from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from src.utils import PlanExecState, PlanSummaryState, StepTaskState
from src.prompt_template import step_system_message, step_human_message, step_input_variables, summary_system_message, summary_human_message, summary_input_variables
from src.utils import StepTaskFormat, PlanSummaryFormat

from dotenv import load_dotenv
from src.llm import create_chat_llm

load_dotenv()

def task_define(state: PlanExecState):
    messages = [
        SystemMessagePromptTemplate.from_template(step_system_message),
        HumanMessagePromptTemplate.from_template(step_human_message),
    ]
    prompt = ChatPromptTemplate(input_variables=step_input_variables, messages=messages)
    llm = create_chat_llm(temperature=0.3)
    structured_llm = llm.with_structured_output(StepTaskFormat)
    chain = prompt | structured_llm

    # check stop or continue
    if len(state["step_output"]) == len(state["plan"]) or (len(state["step_output"]) > 0 and state["step_output"][-1]["success"].lower() == "no"):
        # summary about this plan and then stop
        messages = [
            SystemMessagePromptTemplate.from_template(summary_system_message),
            HumanMessagePromptTemplate.from_template(summary_human_message),
        ]
        prompt = ChatPromptTemplate(input_variables=summary_input_variables, messages=messages)
        llm = create_chat_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(PlanSummaryFormat)

        chain = prompt | structured_llm
        question = state["original_question"]
        plan_steps = state["plan"]
        plan = f"[{', '.join(plan_steps)}]"
        memory = ""
        for id, item in enumerate(state["step_output"]):
            step_q = state["step_question"][id]
            memory += (
                f"Task: {plan_steps[id]}\n"
                f"Question: {step_q['task']}\n"
                f"Answer: {item['answer']}\n"
                f"Confident score: {item['rating']}\n\n"
            )
        output = chain.invoke({
            "question": question, 
            "plan": plan,
            "memory": memory
        })
        output = PlanSummaryState(**output.model_dump())
        return {"plan_summary": output, "stop": True}
    else:
        plan_steps = state["plan"]
        plan = f"[{', '.join(plan_steps)}]"
        cur_step = plan_steps[len(state["step_output"])]
        memory = ""
        for id in range(len(state["step_output"])):
            memory += f"Task: {plan_steps[id]}\nAnswer: {state['step_output'][id]['answer']}\n\n"
        response = chain.invoke({"plan": plan, "cur_step": cur_step, "memory": memory})
        response = StepTaskState(**response.model_dump())
        return {"step_question": [response]}
