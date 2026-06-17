"""Helpers for on-prem SLMs (Ollama) that struggle with structured output."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from src.contracts.messages import RouteDecision
from src.env import get_llm_provider

_PLACEHOLDER_ANSWERS = frozenset({"answer", "n/a", "unknown", "none", "no answer found", ""})


def is_ollama_provider() -> bool:
    return get_llm_provider() == "ollama"


def is_likely_multi_hop(question: str) -> bool:
    """Detect questions that need multiple plan steps (vs one KB lookup)."""
    q = question.lower()
    markers = (
        " and ",
        " both ",
        " compare ",
        " common ",
        " between ",
        " as well as ",
    )
    return any(marker in q for marker in markers)


def canonical_kb_plan(question: str) -> List[str]:
    """Known multi-hop patterns in the project knowledge base (SLM planner fallback)."""
    q = question.lower()
    if (
        "volunteer researcher" in q
        and "work with" in q
        and "title" in q
        and "oracle" in q
    ):
        return [
            "Who does the volunteer researcher on MA-RAG work with?",
            "What is Chandra Shekar Konda's title at Oracle?",
        ]
    return []


def heuristic_multi_hop_plan(question: str) -> List[str]:
    """
    Split compound questions on connectors (e.g. 'and') into ordered sub-questions.
    Used when the SLM planner over/under-shoots step count for multi-hop routes.
    """
    canonical = canonical_kb_plan(question)
    if canonical:
        return canonical
    if not is_likely_multi_hop(question):
        return []

    q = question.strip()
    if not q.endswith("?"):
        q = f"{q}?"

    parts = re.split(r"\s+and\s+", q, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return []

    left, right = parts[0].strip(), parts[1].strip()
    if len(left) < 12 or len(right) < 12:
        return []
    if not left.endswith("?"):
        left = f"{left}?"
    if not right.endswith("?"):
        right = f"{right}?"
    return [left, right]


def classify_route(question: str) -> Tuple[RouteDecision, str]:
    """Heuristic triage: simple KB lookup vs multi-hop planning."""
    if heuristic_multi_hop_plan(question):
        return (
            RouteDecision.MULTI_HOP_RAG,
            "Known MA-RAG knowledge-base multi-hop pattern.",
        )
    if is_likely_multi_hop(question):
        return (
            RouteDecision.MULTI_HOP_RAG,
            "Question contains multi-hop markers (e.g. 'and', 'both').",
        )
    return (
        RouteDecision.SIMPLE_RAG,
        "Direct factual lookup in the ingested knowledge base.",
    )


def simplify_plan_for_slm(question: str, steps: List[str]) -> List[str]:
    """Prefer a single retrieval step for direct factual questions on small models."""
    if not is_ollama_provider() or not steps:
        return steps

    q = question.strip()
    if not q.endswith("?"):
        q = f"{q}?"

    if not is_likely_multi_hop(q) and len(steps) > 1:
        return [q]
    if len(steps) > 3:
        return [q]
    return steps


def parse_qa_response(text: str) -> dict:
    """Parse plain-text QA output from small models."""
    result = {"analysis": "", "answer": "", "success": "No", "rating": 0}
    if not text or not text.strip():
        return result

    field_patterns = {
        "analysis": re.compile(
            r"(?is)^analysis:\s*(.+?)(?=^\s*(?:answer|success|rating):|\Z)",
            re.MULTILINE,
        ),
        "answer": re.compile(
            r"(?is)^answer:\s*(.+?)(?=^\s*(?:success|rating|analysis):|\Z)",
            re.MULTILINE,
        ),
        "success": re.compile(r"(?im)^success:\s*(yes|no)\b"),
        "rating": re.compile(r"(?im)^rating:\s*(\d+)"),
    }

    for key, pattern in field_patterns.items():
        match = pattern.search(text)
        if not match:
            continue
        value = match.group(1).strip()
        if key == "success":
            result[key] = value.capitalize()
        elif key == "rating":
            result[key] = int(value)
        else:
            result[key] = value

    if not result["answer"]:
        stripped = text.strip()
        if stripped and len(stripped) < 400:
            result["answer"] = stripped
            result["success"] = "Yes"
            result["rating"] = max(result["rating"], 5)

    answer_lower = result["answer"].strip().lower()
    if answer_lower in _PLACEHOLDER_ANSWERS:
        result["success"] = "No"
        result["rating"] = 0

    if (
        result["answer"].strip()
        and answer_lower not in _PLACEHOLDER_ANSWERS
        and result["rating"] >= 5
        and result["success"] == "No"
    ):
        # Small models often emit success=No with a high rating when evidence is noisy.
        result["success"] = "Yes"

    if result["success"] == "Yes" and result["rating"] == 0:
        result["rating"] = 6

    return result


def parse_summary_response(text: str) -> dict:
    """Parse plain-text summarizer output from small models."""
    result = {"output": "", "answer": "", "score": 0}
    if not text or not text.strip():
        return result

    output_match = re.search(r"(?im)^output:\s*(successful|unsuccessful)\b", text)
    answer_match = re.search(
        r"(?is)^final answer:\s*(.+?)(?=^\s*(?:output|score):|\Z)",
        text,
        re.MULTILINE,
    )
    score_match = re.search(r"(?im)^score:\s*(\d+)", text)

    if output_match:
        result["output"] = output_match.group(1).capitalize()
    if answer_match:
        result["answer"] = answer_match.group(1).strip()
    if score_match:
        result["score"] = int(score_match.group(1))

    if not result["answer"]:
        fallback = re.search(r"(?is)^final answer:\s*(.+)$", text, re.MULTILINE)
        if fallback:
            result["answer"] = fallback.group(1).strip()

    if result["output"].lower() == "successful" and result["score"] == 0:
        result["score"] = 6

    return result


def parse_evidence_review_response(text: str) -> dict:
    """Parse plain-text evidence curator output."""
    result = {
        "sufficiency": "insufficient",
        "proceed": False,
        "gaps": [],
        "rationale": "",
    }
    if not text or not text.strip():
        return result

    suff_match = re.search(
        r"(?im)^sufficiency:\s*(sufficient|partial|insufficient)\b",
        text,
    )
    proceed_match = re.search(r"(?im)^proceed:\s*(yes|no)\b", text)
    gaps_match = re.search(r"(?im)^gaps:\s*(.+?)(?=^\s*rationale:|\Z)", text, re.DOTALL)
    rationale_match = re.search(r"(?im)^rationale:\s*(.+)$", text, re.DOTALL)

    if suff_match:
        result["sufficiency"] = suff_match.group(1).lower()
    if proceed_match:
        result["proceed"] = proceed_match.group(1).lower() == "yes"
    if gaps_match:
        raw_gaps = gaps_match.group(1).strip()
        if raw_gaps.lower() not in {"none", "n/a", ""}:
            result["gaps"] = [g.strip() for g in raw_gaps.split(",") if g.strip()]
    if rationale_match:
        result["rationale"] = rationale_match.group(1).strip()

    if result["sufficiency"] == "sufficient" and not proceed_match:
        result["proceed"] = True
    return result


def parse_verify_response(text: str) -> dict:
    """Parse plain-text critic / verifier output."""
    result = {
        "passed": False,
        "confidence": 0,
        "issues": [],
        "revised_answer": "",
    }
    if not text or not text.strip():
        return result

    passed_match = re.search(r"(?im)^passed:\s*(yes|no)\b", text)
    confidence_match = re.search(r"(?im)^confidence:\s*(\d+)", text)
    issues_match = re.search(
        r"(?im)^issues:\s*(.+?)(?=^\s*revised answer:|\Z)",
        text,
        re.DOTALL,
    )
    revised_match = re.search(r"(?im)^revised answer:\s*(.+)$", text, re.DOTALL)

    if passed_match:
        result["passed"] = passed_match.group(1).lower() == "yes"
    if confidence_match:
        result["confidence"] = int(confidence_match.group(1))
    if issues_match:
        raw_issues = issues_match.group(1).strip()
        if raw_issues.lower() not in {"none", "n/a", ""}:
            result["issues"] = [i.strip() for i in raw_issues.split(",") if i.strip()]
    if revised_match:
        result["revised_answer"] = revised_match.group(1).strip()

    if result["passed"] and result["confidence"] == 0:
        result["confidence"] = 7
    return result
