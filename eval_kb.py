#!/usr/bin/env python3
"""Smoke-eval MA-RAG agentic workflow against knowledge-base questions."""

from __future__ import annotations

import argparse
import json
import sys
import warnings

warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")
warnings.filterwarnings("ignore", message=".*allowed_objects.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic\.main")

from dotenv import load_dotenv

from src.env import get_local_index_dir
from src.local_retrieval import LocalRetrieverTool, local_index_exists
from src.llm import describe_active_llm
from src.workflow import WorkflowEngine

load_dotenv()

# (question, expected_parts, label, match_mode)
# match_mode: "any" | "all"
KB_CASES = [
    (
        "What is the current completed phase of the MA-RAG prototype?",
        ["Phase 0"],
        "single-hop",
        "any",
    ),
    (
        "Who is the volunteer researcher implementing MA-RAG Phase 0?",
        ["Bhargav Boyapati", "Bhargav"],
        "single-hop",
        "any",
    ),
    (
        "What company is Chandra Shekar Konda the AI Technical Director at?",
        ["Oracle"],
        "single-hop",
        "any",
    ),
    (
        "Who does the volunteer researcher on MA-RAG work with, and what is that person's title at Oracle?",
        ["Chandra Shekar Konda", "AI Technical Director"],
        "multi-hop",
        "all",
    ),
    (
        "What is the next phase after Phase 0 for MA-RAG?",
        ["Phase 1"],
        "single-hop",
        "any",
    ),
    (
        "What is the name of the IEEE task force for this effort?",
        ["Talent Meets AI", "Talent Meets"],
        "single-hop",
        "any",
    ),
    (
        "What technology is used for the local retrieval index?",
        ["FAISS"],
        "single-hop",
        "any",
    ),
    (
        "What is the research system being prototyped in this project?",
        ["MA-RAG"],
        "single-hop",
        "any",
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate agentic MA-RAG on KB questions")
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        type=int,
        help="Run only case number(s) from the built-in list (1-based)",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Write machine-readable results JSON to this path",
    )
    return parser.parse_args()


def _case_passes(answer: str, expected_parts: list[str], match_mode: str) -> bool:
    answer_lower = (answer or "").lower()
    if match_mode == "all":
        return all(part.lower() in answer_lower for part in expected_parts)
    return any(part.lower() in answer_lower for part in expected_parts)


def main() -> int:
    args = parse_args()
    if not local_index_exists():
        print(
            f"No local index at {get_local_index_dir()}\n"
            "Run: python ingest.py ./docs",
            file=sys.stderr,
        )
        return 1

    cases = list(KB_CASES)
    if args.case_ids:
        selected = []
        for case_id in args.case_ids:
            if case_id < 1 or case_id > len(KB_CASES):
                print(f"Invalid --case {case_id}; choose 1-{len(KB_CASES)}", file=sys.stderr)
                return 1
            selected.append(KB_CASES[case_id - 1])
        cases = selected

    print(f"LLM: {describe_active_llm()}", file=sys.stderr)
    engine = WorkflowEngine(LocalRetrieverTool(top_k=3))

    passed = 0
    results = []
    for index, (question, expected_parts, label, match_mode) in enumerate(cases, start=1):
        print(f"\n=== Case {index} ({label}) ===")
        print(f"Q: {question}")
        package = engine.run(question)
        ok = _case_passes(package.answer, expected_parts, match_mode)
        verify_ok = package.verify_passed is not False
        case_pass = ok and verify_ok and bool(package.answer.strip())

        print(f"A: {package.answer}")
        print(f"Expected ({match_mode}): {expected_parts}")
        print(f"Route: {package.route_decision.value if package.route_decision else 'n/a'}")
        print(f"Verify passed: {package.verify_passed}")
        print(f"Plan steps: {len(package.plan_steps)}")
        print(f"Result: {'PASS' if case_pass else 'FAIL'}")

        results.append(
            {
                "case": index,
                "label": label,
                "question": question,
                "answer": package.answer,
                "expected": expected_parts,
                "match_mode": match_mode,
                "route": package.route_decision.value if package.route_decision else None,
                "verify_passed": package.verify_passed,
                "plan_steps": len(package.plan_steps),
                "passed": case_pass,
                "run_id": package.run_id,
            }
        )

        if case_pass:
            passed += 1

    total = len(cases)
    print(f"\n=== Summary: {passed}/{total} passed ===")

    if args.json_out:
        payload = {
            "llm": describe_active_llm(),
            "passed": passed,
            "total": total,
            "results": results,
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"Wrote results to {args.json_out}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
