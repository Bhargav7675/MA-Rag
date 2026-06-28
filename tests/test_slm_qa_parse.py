"""Tests for SLM QA parsing and context fallback."""

from src.slm_helpers import (
    apply_factual_answer_fallback,
    extract_factual_answer_from_context,
    is_parser_artifact_answer,
    parse_qa_response,
)


def test_parser_artifact_success_no():
    assert is_parser_artifact_answer("Success: No")
    parsed = parse_qa_response("Success: No\nRating: 0")
    assert parsed["answer"] == ""
    assert parsed["success"] == "No"


def test_extract_platform_lead_reports_to_manager():
    context = (
        "Platform Lead: Alex Rivera - Platform Lead reports to Jordan Lee "
        "(VP of Engineering) - Alex Rivera maintains the Kubernetes platform "
        "used by Greenfield Health"
    )
    answer = extract_factual_answer_from_context(
        "Who does the platform lead report to?",
        context,
    )
    assert answer == "Jordan Lee"


def test_format_summit_multi_hop():
    from src.contracts.messages import StepAnswer
    from src.workflow.finalize_helpers import format_summit_multi_hop_answer

    steps = [
        StepAnswer(
            step_index=0,
            plan_step="s1",
            task="t1",
            analysis="",
            answer="Jordan Lee",
            success=True,
            confidence=10,
            doc_ids=[],
        ),
        StepAnswer(
            step_index=1,
            plan_step="s2",
            task="t2",
            analysis="",
            answer="Greenfield Health",
            success=True,
            confidence=10,
            doc_ids=[],
        ),
    ]
    answer = format_summit_multi_hop_answer(steps)
    assert "Jordan Lee" in answer
    assert "Greenfield Health" in answer
    assert "Alex Rivera" in answer


def test_extract_cto_from_greenfield_context():
    context = "- **CEO:** Priya Nair - **CTO:** Marcus Chen - **Chief Medical Officer:**"
    answer = extract_factual_answer_from_context(
        "Who is the CTO of Greenfield Health?",
        context,
    )
    assert answer == "Marcus Chen"


def test_factual_fallback_uses_raw_documents_not_only_notes():
    raw = "- **CEO:** Priya Nair - **CTO:** Marcus Chen"
    notes_context = "doc_0: [No CTO information in this passage]"
    parsed = {"analysis": "", "answer": "", "success": "No", "rating": 0}
    out = apply_factual_answer_fallback(
        "Who is the CTO of Greenfield Health?",
        parsed,
        raw_documents=[raw],
        formatted_context=notes_context,
    )
    assert out["answer"] == "Marcus Chen"
    assert out["success"] == "Yes"


def test_extract_kubernetes_platform_manager():
    context = (
        "Platform Lead: Alex Rivera - Alex Rivera maintains the Kubernetes "
        "platform used by Greenfield Health and other enterprise clients."
    )
    answer = extract_factual_answer_from_context(
        "Who manages the Kubernetes platform used by Greenfield Health?",
        context,
    )
    assert answer == "Alex Rivera"


def test_factual_fallback_confirms_slm_name_with_success_no():
    raw = (
        "Alex Rivera maintains the Kubernetes platform used by "
        "Greenfield Health and other enterprise clients."
    )
    parsed = {
        "analysis": "no relevant info",
        "answer": "Alex Rivera",
        "success": "No",
        "rating": 0,
    }
    out = apply_factual_answer_fallback(
        "Who manages the Kubernetes platform used by Greenfield Health?",
        parsed,
        raw_documents=[raw],
    )
    assert out["answer"] == "Alex Rivera"
    assert out["success"] == "Yes"
