#!/usr/bin/env python3
"""Interactive entry point: ask MA-RAG a single question."""

import argparse
import json
import sys
import warnings

from dotenv import load_dotenv

# Quieter terminal output for demos (upstream SSL / LangGraph / Pydantic noise).
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")
warnings.filterwarnings("ignore", message=".*allowed_objects.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic\.main")

from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

from src.llm import create_chat_llm, describe_active_llm
from src.env import get_local_index_dir
from src.local_retrieval import LocalRetrieverTool, local_index_exists
from src.pipeline import (
    build_ma_rag_graph,
    build_retriever_tool,
    dump_json,
    format_graph_output,
    has_retrieval_index,
    run_question,
)
from src.workflow import WorkflowEngine, format_workflow_output
from src.prompt_template import qa_human_message, qa_input_variables, qa_system_message
from src.utils import QAAnswerFormat

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Ask MA-RAG a single question")
    parser.add_argument("question", nargs="?", help="Question to answer")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0], help="GPU ids for FAISS (if enabled)")
    parser.add_argument(
        "--llm-only",
        action="store_true",
        help="Skip retrieval and answer with the configured LLM only",
    )
    parser.add_argument(
        "--retriever",
        choices=["auto", "local", "dpr"],
        default="auto",
        help="Retrieval backend: local index from ingest.py, original DPR index, or auto",
    )
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Show retrieved chunks without calling the LLM",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write full graph state as JSON",
    )
    parser.add_argument(
        "--agentic",
        action="store_true",
        help="Use typed agent workflow (Track B) instead of LangGraph pipeline",
    )
    return parser.parse_args()


def run_llm_only(question: str):
    messages = [
        SystemMessagePromptTemplate.from_template(qa_system_message),
        HumanMessagePromptTemplate.from_template(qa_human_message),
    ]
    prompt = ChatPromptTemplate(input_variables=qa_input_variables, messages=messages)
    llm = create_chat_llm(temperature=0.3)
    chain = prompt | llm.with_structured_output(QAAnswerFormat)
    return chain.invoke({"context": "", "question": question})


def main():
    args = parse_args()
    question = args.question
    if not question:
        question = input("Question: ").strip()
    if not question:
        print("No question provided.", file=sys.stderr)
        sys.exit(1)

    if args.llm_only:
        output = run_llm_only(question)
        print("=== LLM-only answer (retrieval skipped) ===")
        print(output.answer)
        print(f"Success: {output.success}")
        print(f"Confidence: {output.rating}")
        if args.output_json:
            dump_json(args.output_json, output.model_dump())
            print(f"\nWrote full trace to {args.output_json}")
        return

    use_local = args.retriever in {"auto", "local"} and local_index_exists()
    use_dpr = args.retriever in {"auto", "dpr"} and has_retrieval_index()

    if not use_local and not use_dpr:
        local_dir = get_local_index_dir()
        print(
            "No FAISS retrieval index found yet.\n"
            f"To use local documents, add files under ./docs and run: python ingest.py ./docs\n"
            f"Expected local index: {local_dir}\n"
            "To use the original DPR/Wikipedia index, build/add shards under save_embs/gte-ml-base/.\n"
            "For a quick OpenAI smoke test now, run:\n"
            f"  python ask.py {question!r} --llm-only",
            file=sys.stderr,
        )
        sys.exit(1)

    if use_local:
        retriever_tool = LocalRetrieverTool(top_k=3)
    else:
        retriever_tool = build_retriever_tool(gpu_ids=args.gpus)

    print(f"LLM: {describe_active_llm()}", file=sys.stderr)

    if args.retrieve_only:
        docs, doc_ids = retriever_tool(question)
        print("=== Retrieved chunks ===")
        for idx, (doc_id, doc) in enumerate(zip(doc_ids, docs), start=1):
            print(f"\n--- {idx}. {doc_id} ---")
            print(doc[:1000])
        return

    if args.agentic:
        engine = WorkflowEngine(retriever_tool)
        package = engine.run(question)
        print(format_workflow_output(package))
        if args.output_json:
            dump_json(args.output_json, package.model_dump())
            print(f"\nWrote full trace to {args.output_json}")
        return

    graph = build_ma_rag_graph(retriever_tool)
    output = run_question(graph, question)
    print(format_graph_output(output))

    if args.output_json:
        dump_json(args.output_json, output)
        print(f"\nWrote full trace to {args.output_json}")


if __name__ == "__main__":
    main()
