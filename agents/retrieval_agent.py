"""Retrieval agent — FAISS search with typed contracts."""

from __future__ import annotations

from typing import Any

from src.contracts.messages import RetrievedChunk, RetrievalResponse, RetrievalTask

AGENT_ID = "retrieval"
ROLE = "Research / Retrieval — fetch evidence chunks for a sub-question"


class RetrievalAgent:
    def __init__(
        self,
        retriever_tool: Any,
        *,
        agent_id: str = AGENT_ID,
        role: str = ROLE,
    ):
        self.retriever_tool = retriever_tool
        self.agent_id = agent_id
        self.role = role

    def run(self, task: RetrievalTask) -> RetrievalResponse:
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
