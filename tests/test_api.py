"""FastAPI ingress tests (mocked workflow; optional integration)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.server import fastapi_app
from src.contracts.messages import FinalAnswerPackage, RouteDecision, WorkflowStep
from src.local_retrieval import local_index_exists
from src.tools.faiss_retrieve import TOOL_NAME


def _sample_package() -> FinalAnswerPackage:
    return FinalAnswerPackage(
        run_id="testrun01",
        question="What is the current completed phase of the MA-RAG prototype?",
        answer="Phase 0",
        confidence=10,
        plan_steps=["What is the current completed phase of the MA-RAG prototype?"],
        step_answers=[],
        workflow_trace=[
            WorkflowStep.ROUTE,
            WorkflowStep.INIT_PLAN,
            WorkflowStep.RETRIEVE,
            WorkflowStep.GENERATE,
            WorkflowStep.FINALIZE,
            WorkflowStep.VERIFY,
        ],
        chunk_ids_used=["doc#0"],
        verify_passed=True,
        route_decision=RouteDecision.SIMPLE_RAG,
        evidence_ledger_path="/tmp/test.jsonl",
    )


@pytest.fixture
def client() -> TestClient:
    return TestClient(fastapi_app)


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "llm" in payload
    assert payload["index_ready"] == local_index_exists()


def test_ask_returns_503_when_index_missing(client: TestClient):
    with patch("src.api.server.local_index_exists", return_value=False):
        response = client.post("/ask", json={"question": "test question?"})

    assert response.status_code == 503
    assert "ingest.py" in response.json()["detail"]


def test_ask_full_returns_503_when_index_missing(client: TestClient):
    with patch("src.api.server.local_index_exists", return_value=False):
        response = client.post("/ask/full", json={"question": "test question?"})

    assert response.status_code == 503


def test_ask_returns_mocked_answer(client: TestClient):
    mock_service = MagicMock()
    mock_service.ask.return_value = _sample_package()

    with (
        patch("src.api.server.local_index_exists", return_value=True),
        patch("src.api.server.get_workflow_service", return_value=mock_service),
    ):
        response = client.post(
            "/ask",
            json={"question": "What is the current completed phase of the MA-RAG prototype?"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Phase 0"
    assert payload["route"] == "simple_rag"
    assert payload["verify_passed"] is True
    assert payload["run_id"] == "testrun01"
    mock_service.ask.assert_called_once()


def test_ask_rejects_empty_question(client: TestClient):
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_stream_returns_503_when_index_missing(client: TestClient):
    with patch("src.api.server.local_index_exists", return_value=False):
        response = client.post("/ask/stream", json={"question": "test question?"})

    assert response.status_code == 503


def test_ask_stream_emits_sse_events(client: TestClient):
    from src.workflow.events import WorkflowEvent, WorkflowEventKind

    events = [
        WorkflowEvent(
            run_id="stream01",
            kind=WorkflowEventKind.WORKFLOW_START,
            payload={"question": "test?"},
        ),
        WorkflowEvent(
            run_id="stream01",
            kind=WorkflowEventKind.WORKFLOW_COMPLETE,
            payload={"answer": "Phase 0", "confidence": 10, "verify_passed": True, "route": "simple_rag"},
        ),
    ]

    def fake_stream(*_args, **_kwargs):
        yield from events

    mock_service = MagicMock()
    mock_service.ask_stream.side_effect = fake_stream

    with (
        patch("src.api.server.local_index_exists", return_value=True),
        patch("src.api.server.get_workflow_service", return_value=mock_service),
    ):
        response = client.post("/ask/stream", json={"question": "test question?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "FINAL ANSWER" in body
    assert "Phase 0" in body
    assert "workflow_start" not in body  # human format, not raw json kind
    mock_service.ask_stream.assert_called_once()


def test_ask_stream_json_format(client: TestClient):
    from src.workflow.events import WorkflowEvent, WorkflowEventKind

    events = [
        WorkflowEvent(
            run_id="stream01",
            kind=WorkflowEventKind.WORKFLOW_START,
            payload={"question": "test?"},
        ),
    ]

    mock_service = MagicMock()
    mock_service.ask_stream.return_value = iter(events)

    with (
        patch("src.api.server.local_index_exists", return_value=True),
        patch("src.api.server.get_workflow_service", return_value=mock_service),
    ):
        response = client.post(
            "/ask/stream?stream_format=json",
            json={"question": "test question?"},
        )

    assert response.status_code == 200
    assert "workflow_start" in response.text


def test_tools_returns_503_when_index_missing(client: TestClient):
    with patch("src.api.server.local_index_exists", return_value=True), patch(
        "src.api.server.get_workflow_service",
        side_effect=FileNotFoundError("no index"),
    ):
        response = client.get("/tools")

    assert response.status_code == 503


def test_tools_lists_faiss_retrieve(client: TestClient):
    mock_registry = MagicMock()
    mock_tool = MagicMock()
    mock_tool.model_dump.return_value = {
        "name": TOOL_NAME,
        "description": "FAISS search",
        "parameters": [],
        "version": "1.0.0",
    }
    mock_registry.list_tools.return_value = [mock_tool]

    mock_engine = MagicMock()
    mock_engine.tool_registry = mock_registry
    mock_service = MagicMock()
    mock_service.engine = mock_engine

    with patch("src.api.server.get_workflow_service", return_value=mock_service):
        response = client.get("/tools")

    assert response.status_code == 200
    tools = response.json()
    assert len(tools) == 1
    assert tools[0]["name"] == TOOL_NAME


def test_agents_lists_registered_agents(client: TestClient):
    from src.a2a.registry import AgentDescriptor

    mock_bus = MagicMock()
    mock_bus.list_agents.return_value = [
        AgentDescriptor(agent_id="router", role="Router", message_types=["router.request"]),
        AgentDescriptor(agent_id="retrieval", role="Retrieval", message_types=["retrieval.task"]),
    ]
    mock_engine = MagicMock()
    mock_engine.a2a_bus = mock_bus
    mock_service = MagicMock()
    mock_service.engine = mock_engine

    with patch("src.api.server.get_workflow_service", return_value=mock_service):
        response = client.get("/agents")

    assert response.status_code == 200
    agents = response.json()
    assert {agent["agent_id"] for agent in agents} == {"router", "retrieval"}


def test_invoke_tool_unknown_returns_400(client: TestClient):
    mock_registry = MagicMock()
    from src.tools.schemas import ToolCallResult

    mock_registry.invoke.return_value = ToolCallResult(
        tool_name="missing_tool",
        success=False,
        error="Unknown tool: missing_tool",
    )
    mock_engine = MagicMock()
    mock_engine.tool_registry = mock_registry
    mock_service = MagicMock()
    mock_service.engine = mock_engine

    with patch("src.api.server.get_workflow_service", return_value=mock_service):
        response = client.post(
            "/tools/missing_tool/invoke",
            json={"arguments": {"query": "test"}},
        )

    assert response.status_code == 400


@pytest.mark.integration
def test_tools_integration_when_index_present(client: TestClient):
    if not local_index_exists():
        pytest.skip("local FAISS index not built")

    response = client.get("/tools")
    assert response.status_code == 200
    names = {tool["name"] for tool in response.json()}
    assert TOOL_NAME in names


@pytest.mark.integration
def test_tool_invoke_integration_when_index_present(client: TestClient):
    if not local_index_exists():
        pytest.skip("local FAISS index not built")

    response = client.post(
        f"/tools/{TOOL_NAME}/invoke",
        json={"arguments": {"query": "MA-RAG phase", "top_k": 1}},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["output"]["chunks"]
