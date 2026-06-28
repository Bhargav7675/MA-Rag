#!/usr/bin/env python3
"""Ask MA-RAG questions — run `python ask.py` and type your question."""

from __future__ import annotations

import argparse
import os
import sys

from src.runtime_warnings import configure_runtime_warnings

configure_runtime_warnings()

from dotenv import load_dotenv

from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

from src.llm import create_chat_llm, describe_active_llm
from src.env import get_fast_mode, get_local_index_dir, get_retrieval_top_k
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
from src.workflow.event_display import format_event_human
from src.prompt_template import qa_human_message, qa_input_variables, qa_system_message
from src.utils import QAAnswerFormat

load_dotenv()

_INTERACTIVE_EXIT = frozenset({"exit", "quit", "q", ":q"})
_SHELL_COMMAND_PREFIXES = (
    "cat ",
    "ls ",
    "ll ",
    "cd ",
    "pwd",
    "python ",
    "pip ",
    "curl ",
    "grep ",
    "head ",
    "tail ",
    "less ",
    "more ",
    "open ",
    "nano ",
    "vim ",
)


def _looks_like_shell_command(text: str) -> bool:
    lower = text.strip().lower()
    if not lower:
        return False
    if lower.startswith(("./", "/", "~/", "../")):
        return True
    return any(lower.startswith(prefix) for prefix in _SHELL_COMMAND_PREFIXES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ask questions about your ingested documents.",
        epilog=(
            "Examples:\n"
            "  python ask.py\n"
            '  python ask.py "Who is the CTO of Greenfield Health?"'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Optional single question (omit to stay in chat mode)",
    )

    advanced = parser.add_argument_group("advanced (optional)")
    advanced.add_argument(
        "--full",
        action="store_true",
        help="Slower, more thorough answers (disables speed optimizations)",
    )
    advanced.add_argument(
        "--legacy",
        action="store_true",
        help="Use the original LangGraph pipeline instead of agents",
    )
    advanced.add_argument(
        "--quiet",
        action="store_true",
        help="Hide the live step-by-step trace; show the final answer only",
    )
    advanced.add_argument(
        "--verbose",
        action="store_true",
        help="Show LLM and mode details on startup",
    )
    advanced.add_argument("--gpus", nargs="+", type=int, default=[0], help=argparse.SUPPRESS)
    advanced.add_argument(
        "--llm-only",
        action="store_true",
        help="Skip document retrieval (LLM only)",
    )
    advanced.add_argument(
        "--retriever",
        choices=["auto", "local", "dpr"],
        default="auto",
        help=argparse.SUPPRESS,
    )
    advanced.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Show retrieved document chunks only",
    )
    advanced.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write full run trace to a JSON file",
    )
    # Backward compatibility (defaults already match; flags are accepted silently).
    advanced.add_argument("--agentic", action="store_true", help=argparse.SUPPRESS)
    advanced.add_argument("--stream", action="store_true", help=argparse.SUPPRESS)
    advanced.add_argument("--fast", action="store_true", help=argparse.SUPPRESS)
    advanced.add_argument("-i", "--interactive", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _apply_defaults(args) -> None:
    """Good defaults for interactive use; power users opt out with --full."""
    if args.full:
        os.environ["MA_RAG_FAST_MODE"] = "0"
    elif args.fast:
        os.environ["MA_RAG_FAST_MODE"] = "1"
    elif os.getenv("MA_RAG_FAST_MODE") is None:
        os.environ["MA_RAG_FAST_MODE"] = "1"


def run_llm_only(question: str):
    messages = [
        SystemMessagePromptTemplate.from_template(qa_system_message),
        HumanMessagePromptTemplate.from_template(qa_human_message),
    ]
    prompt = ChatPromptTemplate(input_variables=qa_input_variables, messages=messages)
    llm = create_chat_llm(temperature=0.3)
    chain = prompt | llm.with_structured_output(QAAnswerFormat)
    return chain.invoke({"context": "", "question": question})


def _print_shell_command_hint() -> None:
    print(
        "\nThat looks like a terminal command, not a document question.\n"
        "  • Type quit to leave chat, then run it in your shell.\n"
        "  • Or open a second terminal tab.\n"
        "  • Example: cat data/evidence_ledger/b057377c70aa.jsonl\n"
        "    (lowercase .jsonl — use the Run ID from your last answer)\n",
        file=sys.stderr,
    )


def _print_welcome(*, verbose: bool) -> None:
    print("\nMA-RAG — ask your documents anything.")
    print("Type your question below (not in the shell — wait for Question>).")
    print("Type quit when you are done.\n")
    if verbose:
        fast = "on" if get_fast_mode() else "off"
        print(f"LLM: {describe_active_llm()} (fast mode: {fast})\n", file=sys.stderr)


def _read_interactive_question() -> str | None:
    try:
        line = input("Question> ").strip()
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
        return None
    if not line:
        return ""
    if line.lower() in _INTERACTIVE_EXIT:
        return None
    return line


def _run_agentic(
    engine: WorkflowEngine,
    question: str,
    *,
    show_trace: bool,
    output_json: str | None,
) -> None:
    if show_trace:
        def on_event(event):
            text = format_event_human(event)
            if text:
                print(text, end="", flush=True)

        package = engine.run(question, on_event=on_event)
        print(f"\nRun ID: {package.run_id}")
        if package.evidence_ledger_path:
            print(f"Evidence: {package.evidence_ledger_path}")
        if package.a2a_journal_path:
            print(f"A2A journal: {package.a2a_journal_path}")
    else:
        package = engine.run(question)
        print(format_workflow_output(package))

    if output_json:
        dump_json(output_json, package.model_dump())
        print(f"\nWrote full trace to {output_json}")


def _run_langgraph(graph, question: str, *, output_json: str | None) -> None:
    output = run_question(graph, question)
    print(format_graph_output(output))
    if output_json:
        dump_json(output_json, output)
        print(f"\nWrote full trace to {output_json}")


def main():
    args = parse_args()
    _apply_defaults(args)

    interactive = args.interactive or not args.question
    agentic = not args.legacy
    show_trace = agentic and not args.quiet
    if args.stream:
        show_trace = True

    if args.llm_only and interactive:
        print("Chat mode is not supported with --llm-only.", file=sys.stderr)
        sys.exit(1)
    if args.retrieve_only and interactive:
        print("Chat mode is not supported with --retrieve-only.", file=sys.stderr)
        sys.exit(1)

    if args.llm_only:
        question = args.question or input("Question: ").strip()
        if not question:
            print("No question provided.", file=sys.stderr)
            sys.exit(1)
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
            "No document index found yet.\n"
            "1. Put files in ./docs\n"
            "2. Run: python ingest.py\n"
            "3. Run: python ask.py\n"
            f"\n(Index path: {local_dir})",
            file=sys.stderr,
        )
        sys.exit(1)

    if use_local:
        retriever_tool = LocalRetrieverTool(top_k=get_retrieval_top_k())
    else:
        retriever_tool = build_retriever_tool(gpu_ids=args.gpus)

    if args.verbose:
        print(f"LLM: {describe_active_llm()}", file=sys.stderr)

    if args.retrieve_only:
        question = args.question or input("Question: ").strip()
        if not question:
            print("No question provided.", file=sys.stderr)
            sys.exit(1)
        docs, doc_ids = retriever_tool(question)
        print("=== Retrieved chunks ===")
        for idx, (doc_id, doc) in enumerate(zip(doc_ids, docs), start=1):
            print(f"\n--- {idx}. {doc_id} ---")
            print(doc[:1000])
        return

    engine = None
    graph = None
    if agentic:
        engine = WorkflowEngine(retriever_tool, use_a2a=True)
    else:
        graph = build_ma_rag_graph(retriever_tool)

    def answer_one(question: str) -> None:
        if agentic:
            _run_agentic(
                engine,
                question,
                show_trace=show_trace,
                output_json=args.output_json,
            )
        else:
            _run_langgraph(graph, question, output_json=args.output_json)

    if not interactive:
        if _looks_like_shell_command(args.question):
            _print_shell_command_hint()
            sys.exit(1)
        answer_one(args.question)
        return

    _print_welcome(verbose=args.verbose)
    while True:
        question = _read_interactive_question()
        if question is None:
            print("Bye.", file=sys.stderr)
            break
        if not question:
            continue
        if _looks_like_shell_command(question):
            _print_shell_command_hint()
            continue
        print(file=sys.stderr)
        answer_one(question)
        print(file=sys.stderr)


if __name__ == "__main__":
    main()
