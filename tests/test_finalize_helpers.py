"""Tests for multi-step finalize helpers."""

from src.contracts.messages import StepAnswer
from src.workflow.finalize_helpers import (
    draft_contradicts_successful_steps,
    format_kb_multi_hop_answer,
    reconcile_final_answer,
)


def _step(index: int, answer: str) -> StepAnswer:
    return StepAnswer(
        step_index=index,
        plan_step=f"step {index}",
        task=f"task {index}",
        analysis="",
        answer=answer,
        success=True,
        confidence=10,
        doc_ids=[],
    )


def test_format_kb_volunteer_and_partner():
    steps = [
        _step(0, "The volunteer researcher is Bhargav Boyapati."),
        _step(1, "Bhargav works with Chandra Shekar Konda on MA-RAG."),
    ]
    answer = format_kb_multi_hop_answer(steps)
    assert "Bhargav Boyapati" in answer
    assert "Chandra Shekar Konda" in answer


def test_normalize_step_answer_fixes_false_negative():
    from src.workflow.finalize_helpers import normalize_step_answer

    step = _step(1, "Chandra Shekar Konda")
    step = step.model_copy(update={"success": False, "confidence": 0})
    fixed = normalize_step_answer(step)
    assert fixed.success is True
    assert fixed.confidence >= 6


def test_finalize_multi_hop_accepts_grounded_answer():
    from src.workflow.finalize_helpers import finalize_multi_hop_quality

    steps = [
        _step(0, "Bhargav Boyapati"),
        _step(1, "Chandra Shekar Konda"),
    ]
    answer = "Bhargav Boyapati works with Chandra Shekar Konda at Oracle."
    out, conf, passed, issues = finalize_multi_hop_quality(answer, 4, False, steps)
    assert passed is True
    assert conf >= 10
    assert "Chandra" in out


def test_reconcile_replaces_oracle_only_summary():
    steps = [
        _step(0, "Bhargav Boyapati"),
        _step(1, "Chandra Shekar Konda"),
    ]
    bad = "Bhargav Boyapati works at Oracle."
    assert draft_contradicts_successful_steps(bad, steps)
    fixed, issues = reconcile_final_answer(bad, steps)
    assert "Chandra Shekar Konda" in fixed
    assert issues
