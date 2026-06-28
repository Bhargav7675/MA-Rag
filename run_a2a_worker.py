#!/usr/bin/env python3
"""
Remote A2A worker — processes agent requests from the file queue.

Usage (two-terminal fully distributed agentic mode):

  Terminal 1:
    export MA_RAG_A2A_TRANSPORT=file_queue
    python run_a2a_worker.py

  Terminal 2:
    export MA_RAG_A2A_TRANSPORT=file_queue
    python ask.py "your question" --agentic
    # or: python run_api.py  →  POST /ask/stream

In Oracle production, replace the file queue dirs with OCI Queue consumers
without changing A2A envelope shape or agent contracts.
"""

from __future__ import annotations

import signal
import sys
import time

from dotenv import load_dotenv

from src.runtime_warnings import configure_runtime_warnings

configure_runtime_warnings()
load_dotenv()

from src.a2a.file_queue_bus import FileQueueA2ABus, process_pending_queue
from src.env import get_a2a_queue_dir, get_a2a_transport
from src.local_retrieval import LocalRetrieverTool, local_index_exists
from src.workflow.a2a_setup import register_a2a_agents
from src.workflow.engine import WorkflowEngine


def main() -> None:
    if get_a2a_transport() != "file_queue":
        print(
            "Set MA_RAG_A2A_TRANSPORT=file_queue before starting the worker.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not local_index_exists():
        print("Build FAISS index first: python ingest.py ./docs", file=sys.stderr)
        sys.exit(1)

    retriever = LocalRetrieverTool(top_k=3)
    engine = WorkflowEngine(retriever, use_a2a=False)
    bus = FileQueueA2ABus()
    register_a2a_agents(
        bus,
        router=engine.router,
        planner=engine.planner,
        retrieval=engine.retrieval,
        evidence_curator=engine.evidence_curator,
        step_definer=engine.step_definer,
        rag_step=engine.rag_step,
        summarizer=engine.summarizer,
        critic=engine.critic,
    )

    running = True

    def stop(_signum, _frame) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    queue_dir = get_a2a_queue_dir()
    print(f"A2A worker listening on {queue_dir}/pending (Ctrl+C to stop)")

    while running:
        handled = process_pending_queue(bus)
        if handled:
            print(f"Processed {handled} A2A message(s)")
        time.sleep(0.05)


if __name__ == "__main__":
    main()
