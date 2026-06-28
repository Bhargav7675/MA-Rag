"""Human-readable formatting for workflow events and final packages."""

from __future__ import annotations

from typing import Optional

from src.contracts.messages import FinalAnswerPackage
from src.workflow.events import WorkflowEvent, WorkflowEventKind


def format_event_human(event: WorkflowEvent) -> Optional[str]:
    """One clear line (or short block) per meaningful event. Returns None to skip noise."""
    kind = event.kind
    payload = event.payload

    if kind == WorkflowEventKind.AGENT_START:
        return None

    if kind == WorkflowEventKind.WORKFLOW_START:
        question = payload.get("question", "").strip()
        return f"\nQuestion\n  {question}\n"

    if kind == WorkflowEventKind.WORKFLOW_ERROR:
        return f"\nError: {payload.get('error', 'unknown')}\n"

    if kind == WorkflowEventKind.AGENT_COMPLETE:
        agent = (event.agent or "agent").replace("_", " ").title()
        step = event.workflow_step
        step_label = f" ({step})" if step else ""

        if event.agent == "router":
            decision = payload.get("decision", "?")
            rationale = payload.get("rationale", "")
            line = f"[{agent}{step_label}] {decision}"
            if rationale:
                line += f" — {rationale}"
            return line + "\n"

        if event.agent == "planner":
            steps = payload.get("steps") or []
            lines = [f"[{agent}{step_label}] {len(steps)} plan step(s):"]
            for index, plan_step in enumerate(steps, start=1):
                lines.append(f"  {index}. {plan_step}")
            return "\n".join(lines) + "\n"

        if event.agent == "step_definer":
            task = payload.get("task", "")
            return f"[{agent}] Task: {task}\n"

        if event.agent == "retrieval":
            index = payload.get("step_index", 0) + 1
            count = payload.get("chunk_count", 0)
            return f"[{agent}] Step {index} — retrieved {count} chunk(s)\n"

        if event.agent == "evidence_curator":
            index = payload.get("step_index", 0) + 1
            suff = payload.get("sufficiency", "?")
            proceed = "proceed" if payload.get("proceed") else "stop"
            return f"[{agent}] Step {index} — {suff}, {proceed}\n"

        if event.agent == "rag_step":
            index = payload.get("step_index", 0) + 1
            answer = (payload.get("answer") or "").strip()
            success = payload.get("success")
            if success is None:
                status = "ok" if answer else "failed"
            else:
                status = "ok" if success or answer else "failed"
            line = f"[{agent}] Step {index} — {status}"
            if answer:
                preview = answer[:160] + ("…" if len(answer) > 160 else "")
                line += f"\n  → {preview}"
            return line + "\n"

        if event.agent == "summarizer":
            answer = payload.get("draft_answer") or payload.get("answer", "")
            preview = answer[:120] + ("…" if len(answer) > 120 else "")
            return f"[{agent}] Draft: {preview}\n"

        if event.agent == "critic":
            passed = "passed" if payload.get("passed") else "failed"
            issues = payload.get("issues") or []
            line = f"[{agent}] Verify {passed}"
            if issues:
                line += f" — {', '.join(issues)}"
            return line + "\n"

        return f"[{agent}{step_label}] done\n"

    if kind == WorkflowEventKind.WORKFLOW_COMPLETE:
        answer = payload.get("answer", "").strip()
        confidence = payload.get("confidence", "?")
        route = payload.get("route", "?")
        verified = payload.get("verify_passed")
        verify_label = (
            "passed" if verified is True else "failed" if verified is False else "n/a"
        )
        return (
            "\n"
            + "=" * 56
            + "\n"
            + "FINAL ANSWER\n"
            + "=" * 56
            + "\n"
            + answer
            + "\n\n"
            + f"Confidence : {confidence}\n"
            + f"Verify     : {verify_label}\n"
            + f"Route      : {route}\n"
            + f"Run ID     : {event.run_id}\n"
        )

    return None


def format_workflow_output_clear(package: FinalAnswerPackage) -> str:
    """Clean CLI / demo output — answer first, details after."""
    lines: list[str] = [
        "",
        "=" * 56,
        "FINAL ANSWER",
        "=" * 56,
        package.answer,
        "",
        f"Confidence : {package.confidence}",
    ]

    if package.route_decision is not None:
        lines.append(f"Route      : {package.route_decision.value}")
    if package.verify_passed is not None:
        lines.append(
            f"Verify     : {'passed' if package.verify_passed else 'failed'}"
        )
    if package.verify_issues:
        lines.append(f"Issues     : {', '.join(package.verify_issues)}")

    lines.extend(["", "-" * 56, "PLAN", "-" * 56])
    for index, step in enumerate(package.plan_steps, start=1):
        lines.append(f"  {index}. {step}")

    if package.step_answers:
        lines.extend(["", "-" * 56, "STEP ANSWERS", "-" * 56])
        for step in package.step_answers:
            lines.append(f"\n  Step {step.step_index + 1}: {step.task}")
            lines.append(f"  → {step.answer}")

    lines.extend(
        [
            "",
            "-" * 56,
            "TRACE",
            "-" * 56,
            f"  {', '.join(s.value for s in package.workflow_trace)}",
            f"  Run ID: {package.run_id}",
        ]
    )
    if package.evidence_ledger_path:
        lines.append(f"  Evidence: {package.evidence_ledger_path}")
    if package.a2a_journal_path:
        lines.append(f"  A2A journal: {package.a2a_journal_path}")

    return "\n".join(lines)
