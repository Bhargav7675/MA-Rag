"""Retrieval agent — FAISS search via MCP-style faiss_retrieve tool."""

from __future__ import annotations

from typing import Any, Optional

from src.contracts.messages import RetrievedChunk, RetrievalResponse, RetrievalTask
from src.tools.faiss_retrieve import TOOL_NAME
from src.tools.registry import ToolRegistry
from src.tools.schemas import ToolCallRequest

AGENT_ID = "retrieval"
ROLE = "Research / Retrieval — fetch evidence chunks for a sub-question"


class RetrievalAgent:
    def __init__(
        self,
        retriever_tool: Any = None,
        *,
        tool_registry: Optional[ToolRegistry] = None,
        agent_id: str = AGENT_ID,
        role: str = ROLE,
    ):
        self.retriever_tool = retriever_tool
        self.tool_registry = tool_registry
        self.agent_id = agent_id
        self.role = role

    def run(self, task: RetrievalTask) -> RetrievalResponse:
        if self.tool_registry is not None:
            return self._run_via_tool(task)
        if self.retriever_tool is None:
            raise RuntimeError("RetrievalAgent requires tool_registry or retriever_tool")
        docs, doc_ids = self.retriever_tool(task.question)
        chunks = [
            RetrievedChunk(doc_id=doc_id, text=doc)
            for doc_id, doc in zip(doc_ids, docs)
        ]
        return RetrievalResponse(
            run_id=task.run_id,
            step_index=task.step_index,
            question=task.question,
            chunks=chunks,
        )

    def _run_via_tool(self, task: RetrievalTask) -> RetrievalResponse:
        result = self.tool_registry.invoke(
            ToolCallRequest(
                tool_name=TOOL_NAME,
                arguments={"query": task.question, "top_k": task.top_k},
                run_id=task.run_id,
                caller_agent=self.agent_id,
            )
        )
        if not result.success:
            return RetrievalResponse(
                run_id=task.run_id,
                step_index=task.step_index,
                question=task.question,
                chunks=[],
            )

        raw_chunks = (result.output or {}).get("chunks") or []
        chunks = [
            RetrievedChunk(
                doc_id=item["doc_id"],
                text=item["text"],
                score=item.get("score"),
            )
            for item in raw_chunks
        ]
        return RetrievalResponse(
            run_id=task.run_id,
            step_index=task.step_index,
            question=task.question,
            chunks=chunks,
        )
