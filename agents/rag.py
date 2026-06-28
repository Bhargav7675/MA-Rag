from src.utils import RagState, QAAnswerFormat, QAAnswerState
from src.prompt_template import extract_system_messgage, extract_human_message, extract_input_variables
from src.prompt_template import (
    qa_grounded_addon,
    qa_human_message,
    qa_input_variables,
    qa_slm_output_format_addon,
    qa_system_message,
)
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from src.llm import create_chat_llm, is_agent_ollama
from langchain_core.output_parsers import StrOutputParser
from src.slm_helpers import apply_factual_answer_fallback, parse_qa_response
from langgraph.graph import StateGraph, START, END
from src.env import get_skip_rag_extract


def _raw_passage_notes(documents: list[str]) -> list[str]:
    """Use retrieved text directly — no per-chunk extract LLM calls."""
    return [f"[{doc[:1500]}]" for doc in documents]


def _extract_notes(question: str, documents: list[str]) -> list[str]:
    if get_skip_rag_extract():
        return _raw_passage_notes(documents)
    messages = [
        SystemMessagePromptTemplate.from_template(extract_system_messgage),
        HumanMessagePromptTemplate.from_template(extract_human_message),
    ]
    prompt = ChatPromptTemplate(input_variables=extract_input_variables, messages=messages)
    llm = create_chat_llm(agent_id="rag_step", temperature=0.0)
    chain = prompt | llm | StrOutputParser()
    list_notes = []
    for doc in documents:
        note = chain.invoke({"passage": doc, "question": question})
        list_notes.append(f"[{note}]")
    return list_notes


def _generate_answer(
    question: str,
    doc_ids: list,
    notes: list[str],
    *,
    raw_documents=None,
) -> QAAnswerState:
    tmps = []
    for doc_id, note in zip(doc_ids, notes):
        tmps.append(f"doc_{doc_id}: {note}")
    docs = "\n\n".join(tmps)
    system_message = qa_system_message + qa_grounded_addon
    if is_agent_ollama("rag_step"):
        system_message += qa_slm_output_format_addon
    messages = [
        SystemMessagePromptTemplate.from_template(system_message),
        HumanMessagePromptTemplate.from_template(qa_human_message),
    ]
    prompt = ChatPromptTemplate(input_variables=qa_input_variables, messages=messages)
    llm = create_chat_llm(
        agent_id="rag_step",
        temperature=0.0 if is_agent_ollama("rag_step") else 0.3,
    )
    inputs = {"context": docs, "question": question}
    if is_agent_ollama("rag_step"):
        text = (prompt | llm | StrOutputParser()).invoke(inputs)
        parsed = parse_qa_response(text)
        parsed = apply_factual_answer_fallback(
            question,
            parsed,
            raw_documents=raw_documents,
            formatted_context=docs,
        )
        return QAAnswerState(**parsed)
    structured_llm = llm.with_structured_output(QAAnswerFormat)
    raw = (prompt | structured_llm).invoke(inputs)
    return QAAnswerState(**raw.model_dump())


def run_rag_extract_generate(
    question: str,
    documents: list[str],
    doc_ids: list,
) -> dict:
    """Extract + generate on pre-retrieved documents (workflow evidence path)."""
    notes = _extract_notes(question, documents)
    answer = _generate_answer(question, doc_ids, notes, raw_documents=documents)
    return {
        "question": question,
        "documents": documents,
        "doc_ids": doc_ids,
        "notes": notes,
        "final_raw_answer": answer,
    }


def build_rag_agent(retriever_tool=None):
    def retrieve(state: RagState):
        user_question = state["question"]
        list_docs, list_doc_ids = retriever_tool(query=user_question)
        state["documents"] = list_docs
        state["doc_ids"] = list_doc_ids
        return state

    def extract(state: RagState):
        state["notes"] = _extract_notes(state["question"], state["documents"])
        return state

    def generate(state: RagState):
        response = _generate_answer(
            state["question"],
            state["doc_ids"],
            state["notes"],
            raw_documents=state.get("documents"),
        )
        return {"final_raw_answer": response}

    rag_graph_builder = StateGraph(RagState)
    rag_graph_builder.add_node("retrieve", retrieve)
    rag_graph_builder.add_node("extract", extract)
    rag_graph_builder.add_node("generate", generate)

    rag_graph_builder.add_edge(START, "retrieve")
    rag_graph_builder.add_edge("retrieve", "extract")
    rag_graph_builder.add_edge("extract", "generate")
    rag_graph_builder.add_edge("generate", END)
    rag_graph = rag_graph_builder.compile()
    return rag_graph
