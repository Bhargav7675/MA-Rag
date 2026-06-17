"""Unit tests for SLM planning heuristics."""

from src.slm_helpers import heuristic_multi_hop_plan, is_likely_multi_hop


def test_heuristic_splits_and_questions():
    question = (
        "Who does the volunteer researcher on MA-RAG work with, "
        "and what is that person's title at Oracle?"
    )
    steps = heuristic_multi_hop_plan(question)
    assert len(steps) == 2
    assert "work with" in steps[0].lower()
    assert "title" in steps[1].lower()


def test_heuristic_returns_empty_for_single_hop():
    question = "What is the current completed phase of the MA-RAG prototype?"
    assert heuristic_multi_hop_plan(question) == []
    assert not is_likely_multi_hop(question)
