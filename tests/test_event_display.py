"""Tests for human-readable workflow display."""

from src.contracts.messages import FinalAnswerPackage, RouteDecision, WorkflowStep
from src.workflow.event_display import format_event_human, format_workflow_output_clear
from src.workflow.events import WorkflowEvent, WorkflowEventKind


def test_human_workflow_complete_block():
    event = WorkflowEvent(
        run_id="abc123",
        kind=WorkflowEventKind.WORKFLOW_COMPLETE,
        payload={
            "answer": "Phase 0",
            "confidence": 10,
            "verify_passed": True,
            "route": "simple_rag",
        },
    )
    text = format_event_human(event)
    assert text is not None
    assert "FINAL ANSWER" in text
    assert "Phase 0" in text
    assert "abc123" in text


def test_human_skips_agent_start():
    event = WorkflowEvent(
        run_id="x",
        kind=WorkflowEventKind.AGENT_START,
        agent="router",
    )
    assert format_event_human(event) is None


def test_clear_cli_output_puts_answer_first():
    package = FinalAnswerPackage(
        run_id="r1",
        question="Q?",
        answer="Phase 0",
        confidence=10,
        plan_steps=["step one"],
        step_answers=[],
        workflow_trace=[WorkflowStep.ROUTE, WorkflowStep.FINALIZE],
        route_decision=RouteDecision.SIMPLE_RAG,
    )
    text = format_workflow_output_clear(package)
    assert text.index("FINAL ANSWER") < text.index("PLAN")
    assert "Phase 0" in text
