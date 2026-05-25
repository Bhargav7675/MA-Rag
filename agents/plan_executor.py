from agents.step_definer import task_define
from agents.rag import build_rag_agent
from src.utils import PlanExecState, QAAnswerFormat, QAAnswerState
from src.prompt_template import aggregate_human_message, aggregate_input_variables, aggregate_system_message
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from src.llm import create_chat_llm
from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv

load_dotenv()

def build_plan_executor(retriever_tool = None):
    rag_agent = build_rag_agent(retriever_tool=retriever_tool)

    def single_task_execute(state: PlanExecState):
        cur_task = state["step_question"][-1]
        query = cur_task["task"]
        if cur_task["type"] == "aggregate":
            messages = [
                SystemMessagePromptTemplate.from_template(aggregate_system_message),
                HumanMessagePromptTemplate.from_template(aggregate_human_message),
            ]
            prompt = ChatPromptTemplate(input_variables=aggregate_input_variables, messages=messages)
            llm = create_chat_llm(temperature=0.0)
            structured_llm = llm.with_structured_output(QAAnswerFormat)
            chain = prompt | structured_llm
            response = chain.invoke({"question": query})
            response = QAAnswerState(**response.model_dump())
            step_doc_ids = []
            step_notes = []
        else:
            response = rag_agent.invoke({
                "question": query
            })
            step_doc_ids = [response["doc_ids"]]
            step_notes = [response["notes"]]
            response = response["final_raw_answer"]
    
        return {"step_output": [response], "step_docs_ids": step_doc_ids, "step_notes": step_notes}
    
    def task_definer_out(state: PlanExecState):
        if state["stop"] == True:
            return END
        else:
            return "single_task_execute"
        
    graph_builder = StateGraph(PlanExecState)
    
    graph_builder.add_node("task_definer", task_define)
    graph_builder.add_node("single_task_execute", single_task_execute)
    graph_builder.add_edge(START, "task_definer")
    graph_builder.add_edge("single_task_execute", "task_definer")
    graph_builder.add_conditional_edges("task_definer", task_definer_out)
    graph = graph_builder.compile()
    return graph
