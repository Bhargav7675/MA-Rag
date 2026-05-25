"""Shared graph and retriever setup for MA-RAG entry points."""

from __future__ import annotations

import json
from typing import Any

import torch
from langgraph.graph import END, START, StateGraph
from transformers import AutoModel, AutoTokenizer

from agents.plan import plan_agent
from agents.plan_executor import build_plan_executor
from corpus.retrieve import Retriever
from src.env import get_embedding_model_name, get_index_dataset_name, get_index_dir
from src.utils import GraphState, RetrieveTopChunk, load_corpus


def has_retrieval_index() -> bool:
    index_dir = get_index_dir()
    dataset_name = get_index_dataset_name()
    return index_dir.exists() and any(index_dir.glob(f"{dataset_name}*"))


def resolve_torch_device(gpu_id: int = 0) -> str:
    override = __import__("os").getenv("MA_RAG_DEVICE")
    if override:
        return override
    if torch.cuda.is_available():
        return f"cuda:{gpu_id}"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_retriever_tool(*, gpu_ids: list[int] | None = None):
    gpu_ids = gpu_ids or [0]
    index_dir = get_index_dir()
    dataset_name = get_index_dataset_name()
    if not has_retrieval_index():
        raise FileNotFoundError(
            f"No FAISS index shards matching '{dataset_name}*' under {index_dir}.\n"
            "Build embeddings first — see data/README.md."
        )

    device = resolve_torch_device(gpu_ids[0])
    model_name = get_embedding_model_name()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device)

    retrieve = Retriever(gpu_ids=gpu_ids)
    retrieve.init_index_and_add(root_dir=str(index_dir), dataset_name=dataset_name)

    corpus = load_corpus()
    return RetrieveTopChunk(
        tokenizer=tokenizer,
        embedding_model=model,
        retrieval_model=retrieve,
        corpus=corpus,
    )


def build_ma_rag_graph(retriever_tool):
    plan_executor_agent = build_plan_executor(retriever_tool=retriever_tool)

    def plan_executor_node(state: GraphState):
        output = plan_executor_agent.invoke(
            {
                "original_question": state["original_question"],
                "plan": state["plan"],
                "stop": False,
            }
        )
        return {"past_exp": [output]}

    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("planer_node", plan_agent)
    graph_builder.add_node("plan_executor_node", plan_executor_node)
    graph_builder.add_edge(START, "planer_node")
    graph_builder.add_edge("planer_node", "plan_executor_node")
    graph_builder.add_edge("plan_executor_node", END)
    return graph_builder.compile()


def normalize_question(question: str) -> str:
    q = question.strip()
    if not q.endswith("?"):
        q = f"{q}?"
    return q


def format_graph_output(output: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=== Plan ===")
    for i, step in enumerate(output.get("plan", []), start=1):
        lines.append(f"{i}. {step}")

    past_exp = output.get("past_exp") or []
    if not past_exp:
        lines.append("\n(No execution trace returned.)")
        return "\n".join(lines)

    exp = past_exp[-1]
    summary = exp.get("plan_summary") or {}
    lines.append("\n=== Step trace ===")
    for i, step_q in enumerate(exp.get("step_question", []), start=1):
        step_out = exp["step_output"][i - 1] if i - 1 < len(exp.get("step_output", [])) else {}
        lines.append(f"\n--- Step {i} ({step_q.get('type', 'unknown')}) ---")
        lines.append(f"Task: {step_q.get('task', '')}")
        if step_out:
            lines.append(f"Answer: {step_out.get('answer', '')}")
            lines.append(f"Success: {step_out.get('success', '')}")
            lines.append(f"Confidence: {step_out.get('rating', '')}")

    lines.append("\n=== Final answer ===")
    lines.append(summary.get("answer", "(none)"))
    lines.append(f"Confidence score: {summary.get('score', '(none)')}")
    if summary.get("output"):
        lines.append(f"\nSummary:\n{summary['output']}")
    return "\n".join(lines)


def run_question(graph, question: str) -> dict[str, Any]:
    return graph.invoke({"original_question": normalize_question(question)})


def dump_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
