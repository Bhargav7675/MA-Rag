"""FastAPI ingress for MA-RAG agentic workflow."""

from __future__ import annotations

from src.runtime_warnings import configure_runtime_warnings

configure_runtime_warnings()

import json
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

from src.api.schemas import AskRequest, AskResponse, HealthResponse, ToolInvokeRequest
from src.contracts.messages import FinalAnswerPackage
from src.env import get_api_host, get_api_port, get_local_index_dir
from src.llm import describe_active_llm
from src.local_retrieval import local_index_exists
from src.tools.schemas import ToolCallRequest
from src.workflow.service import clear_workflow_service_cache, get_workflow_service

load_dotenv()

fastapi_app = FastAPI(
    title="MA-RAG API",
    description="Agentic MA-RAG ingress with human-readable /ask/stream SSE.",
    version="0.2.0",
)


@fastapi_app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        llm=describe_active_llm(),
        index_ready=local_index_exists(),
    )


@fastapi_app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    if not local_index_exists():
        raise HTTPException(
            status_code=503,
            detail=f"No local FAISS index at {get_local_index_dir()}. Run: python ingest.py ./docs",
        )
    try:
        service = get_workflow_service()
        package = service.ask(
            request.question,
            run_id=request.run_id,
            metadata=request.metadata,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AskResponse.from_package(package)


@fastapi_app.post("/ask/stream")
def ask_stream(
    request: AskRequest,
    stream_format: Optional[str] = Query(
        default=None,
        description="human (default, readable) or json (machine)",
    ),
) -> StreamingResponse:
    """Server-Sent Events — human-readable agent trace by default."""
    if not local_index_exists():
        raise HTTPException(
            status_code=503,
            detail=f"No local FAISS index at {get_local_index_dir()}. Run: python ingest.py ./docs",
        )

    fmt = stream_format or request.stream_format
    if fmt not in {"human", "json"}:
        raise HTTPException(status_code=422, detail="stream_format must be 'human' or 'json'")

    def event_generator():
        service = get_workflow_service()
        try:
            for event in service.ask_stream(
                request.question,
                run_id=request.run_id,
                metadata=request.metadata,
            ):
                line = event.to_sse(stream_format=fmt)
                if line:
                    yield line
        except Exception as exc:
            if fmt == "json":
                payload = json.dumps({"error": str(exc)})
                yield f"data: {payload}\n\n"
            else:
                yield f"data: Error: {exc}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@fastapi_app.post("/ask/full", response_model=FinalAnswerPackage)
def ask_full(request: AskRequest) -> FinalAnswerPackage:
    if not local_index_exists():
        raise HTTPException(
            status_code=503,
            detail=f"No local FAISS index at {get_local_index_dir()}. Run: python ingest.py ./docs",
        )
    service = get_workflow_service()
    return service.ask(
        request.question,
        run_id=request.run_id,
        metadata=request.metadata,
    )


@fastapi_app.get("/tools")
def list_tools() -> list[dict[str, Any]]:
    try:
        service = get_workflow_service()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return [tool.model_dump() for tool in service.engine.tool_registry.list_tools()]


@fastapi_app.post("/tools/{tool_name}/invoke")
def invoke_tool(tool_name: str, request: ToolInvokeRequest) -> dict[str, Any]:
    service = get_workflow_service()
    result = service.engine.tool_registry.invoke(
        ToolCallRequest(
            tool_name=tool_name,
            arguments=request.arguments,
            run_id=request.run_id,
            caller_agent=request.caller_agent,
        )
    )
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error or "Tool invocation failed")
    return result.model_dump()


@fastapi_app.get("/agents")
def list_agents() -> list[dict[str, Any]]:
    try:
        service = get_workflow_service()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    bus = service.engine.a2a_bus
    if bus is None:
        return []
    return [agent.model_dump() for agent in bus.list_agents()]


def main() -> None:
    import uvicorn

    clear_workflow_service_cache()
    print("MA-RAG API 0.2.0 — /ask/stream defaults to human-readable SSE")
    uvicorn.run(
        "src.api.server:fastapi_app",
        host=get_api_host(),
        port=get_api_port(),
        reload=False,
    )


if __name__ == "__main__":
    main()
