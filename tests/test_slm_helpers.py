"""Unit tests for SLM planning heuristics."""

from src.slm_helpers import (
    apply_factual_answer_fallback,
    canonical_kb_plan,
    extract_factual_answer_from_context,
    heuristic_multi_hop_plan,
    is_likely_multi_hop,
)


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


def test_canonical_plan_volunteer_and_oracle_without_title():
    question = (
        "Who is the volunteer researcher and who do they work with at Oracle?"
    )
    steps = canonical_kb_plan(question)
    assert len(steps) == 2
    assert "volunteer researcher" in steps[0].lower()
    assert "work with" in steps[1].lower()


def test_faiss_extract_from_context():
    question = "What technology is used for the local retrieval index?"
    context = "Retrieval index technology: FAISS (local_index/)"
    assert extract_factual_answer_from_context(question, context) == "FAISS"


def test_faiss_fallback_over_plain_text_slop():
    question = "What technology is used for the local retrieval index?"
    context = "The stack uses FAISS for local_index/"
    parsed = apply_factual_answer_fallback(
        question,
        {"answer": "Plain-text indexing", "success": "Yes", "rating": 7},
        raw_documents=[context],
    )
    assert parsed["answer"] == "FAISS"
