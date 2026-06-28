"""Workflow service — shared entry for CLI and API."""

from __future__ import annotations

import queue
import threading
from functools import lru_cache
from typing import Any, Iterator, Optional

from src.contracts.messages import FinalAnswerPackage
from src.env import get_local_index_dir
from src.local_retrieval import LocalRetrieverTool, local_index_exists
from src.tools import ToolRegistry, default_tool_registry
from src.workflow.engine import WorkflowEngine
from src.workflow.events import WorkflowEvent, WorkflowEventKind


class WorkflowService:
    def __init__(
        self,
        retriever_tool: Any,
        *,
        tool_registry: Optional[ToolRegistry] = None,
        use_a2a: bool = True,
    ):
        self.retriever_tool = retriever_tool
        self._registry = tool_registry or default_tool_registry(retriever_tool)
        self.engine = WorkflowEngine(
            retriever_tool,
            tool_registry=self._registry,
            use_a2a=use_a2a,
        )

    def ask(
        self,
        question: str,
        *,
        run_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        on_event: Optional[Any] = None,
    ) -> FinalAnswerPackage:
        _ = metadata  # reserved for future ingress / tenant context
        return self.engine.run(question, run_id=run_id, on_event=on_event)

    def ask_stream(
        self,
        question: str,
        *,
        run_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Iterator[WorkflowEvent]:
        """
        Run the agentic workflow and yield realtime events as each agent completes.

        Used by POST /ask/stream (SSE) for live IEEE/Oracle-style observability.
        """
        _ = metadata
        event_queue: queue.Queue[WorkflowEvent | None] = queue.Queue()

        def on_event(event: WorkflowEvent) -> None:
            event_queue.put(event)

        error_holder: list[BaseException] = []

        def run_workflow() -> None:
            try:
                self.engine.run(question, run_id=run_id, on_event=on_event)
            except BaseException as exc:
                error_holder.append(exc)
            finally:
                event_queue.put(None)

        thread = threading.Thread(target=run_workflow, daemon=True)
        thread.start()

        while True:
            event = event_queue.get()
            if event is None:
                break
            yield event

        thread.join()
        if error_holder:
            exc = error_holder[0]
            run = run_id or "unknown"
            yield WorkflowEvent(
                run_id=run,
                kind=WorkflowEventKind.WORKFLOW_ERROR,
                payload={"error": str(exc)},
            )
            raise exc

    @property
    def a2a_bus(self):
        return self.engine.a2a_bus

    @property
    def tool_registry(self) -> ToolRegistry:
        return self.engine.tool_registry


@lru_cache(maxsize=1)
def get_workflow_service(*, top_k: int = 3, use_a2a: bool = True) -> WorkflowService:
    if not local_index_exists():
        raise FileNotFoundError(
            f"No local index at {get_local_index_dir()}. Run: python ingest.py ./docs"
        )
    retriever = LocalRetrieverTool(top_k=top_k)
    return WorkflowService(retriever, use_a2a=use_a2a)


def run_ask(
    question: str,
    *,
    run_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> FinalAnswerPackage:
    return get_workflow_service().ask(question, run_id=run_id, metadata=metadata)


def clear_workflow_service_cache() -> None:
    get_workflow_service.cache_clear()
