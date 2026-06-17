"""MCP-style wrapper around the local FAISS retriever."""

from __future__ import annotations

import time
from typing import Any, Optional

from src.contracts.messages import RetrievedChunk
from src.local_retrieval import LocalRetrieverTool
from src.tools.schemas import (
    RetrievedChunkResult,
    ToolCallRequest,
    ToolCallResult,
    ToolDefinition,
    ToolParameter,
)

TOOL_NAME = "faiss_retrieve"


class FaissRetrieveTool:
    """Formal tool interface for local FAISS top-k retrieval."""

    def __init__(self, retriever: Optional[LocalRetrieverTool] = None, *, top_k: int = 3):
        self._retriever = retriever or LocalRetrieverTool(top_k=top_k)
        self._default_top_k = top_k

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=TOOL_NAME,
            description=(
                "Search the local FAISS document index and return top-k evidence chunks "
                "with similarity scores."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Natural-language search query or sub-question.",
                    required=True,
                ),
                ToolParameter(
                    name="top_k",
                    type="integer",
                    description="Number of chunks to return.",
                    required=False,
                    default=self._default_top_k,
                ),
            ],
        )

    def invoke(self, request: ToolCallRequest) -> ToolCallResult:
        started = time.perf_counter()
        try:
            query = str(request.arguments.get("query", "")).strip()
            if not query:
                return ToolCallResult(
                    tool_name=TOOL_NAME,
                    success=False,
                    error="Missing required argument: query",
                )

            top_k = int(request.arguments.get("top_k") or self._default_top_k)
            hits = self._retriever.search_with_scores(query, top_k=top_k)
            chunks = [
                RetrievedChunkResult(
                    doc_id=doc_id,
                    text=text,
                    score=score,
                    source=doc_id.split("#", 1)[0] if "#" in doc_id else doc_id,
                )
                for doc_id, text, score in hits
            ]
            return ToolCallResult(
                tool_name=TOOL_NAME,
                success=True,
                output={"chunks": [chunk.model_dump() for chunk in chunks]},
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            return ToolCallResult(
                tool_name=TOOL_NAME,
                success=False,
                error=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
            )

    def to_retrieved_chunks(self, result: ToolCallResult) -> list[RetrievedChunk]:
        if not result.success or not result.output:
            return []
        raw_chunks = result.output.get("chunks") or []
        return [
            RetrievedChunk(
                doc_id=item["doc_id"],
                text=item["text"],
                score=item.get("score"),
            )
            for item in raw_chunks
        ]


def build_faiss_retrieve_tool(retriever: Any, *, top_k: int = 3) -> FaissRetrieveTool:
    """Accept LocalRetrieverTool or any object with search_with_scores."""
    if isinstance(retriever, FaissRetrieveTool):
        return retriever
    if isinstance(retriever, LocalRetrieverTool):
        return FaissRetrieveTool(retriever, top_k=top_k)
    if hasattr(retriever, "search_with_scores"):
        wrapper = FaissRetrieveTool(top_k=top_k)
        wrapper._retriever = retriever
        return wrapper
    # Legacy callable (query) -> (docs, doc_ids)
    legacy = retriever

    class _LegacyAdapter:
        def search_with_scores(self, query: str, *, top_k: int = 3):
            docs, doc_ids = legacy(query)
            return [(doc_id, doc, None) for doc_id, doc in zip(doc_ids, docs)]

    return FaissRetrieveTool(_LegacyAdapter(), top_k=top_k)
