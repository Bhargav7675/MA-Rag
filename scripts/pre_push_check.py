#!/usr/bin/env python3
"""
Run all checks before pushing to Git.

Usage:
  python scripts/pre_push_check.py           # full (includes eval_kb + Ollama)
  python scripts/pre_push_check.py --quick     # pytest + file-type ingest only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Load project modules after chdir
sys.path.insert(0, str(ROOT))


def _header(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def _run(cmd: list[str], *, cwd: Path | None = None) -> int:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd or ROOT)
    return result.returncode


def check_ollama() -> tuple[bool, str]:
    try:
        import urllib.request

        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as resp:
            if resp.status != 200:
                return False, f"Ollama returned HTTP {resp.status}"
        return True, "Ollama is running"
    except Exception as exc:
        return False, f"Ollama not reachable: {exc}"


def check_llm_config() -> tuple[bool, str]:
    from src.llm import describe_active_llm

    return True, describe_active_llm()


def check_docs_index() -> tuple[bool, str]:
    from src.env import get_local_index_dir
    from src.local_retrieval import local_index_exists

    index_dir = get_local_index_dir()
    if not local_index_exists(index_dir):
        return False, f"No index at {index_dir} — run: python ingest.py"
    chunks = index_dir / "chunks.jsonl"
    count = sum(1 for _ in chunks.open(encoding="utf-8")) if chunks.exists() else 0
    return True, f"Index ready at {index_dir} ({count} chunks)"


def check_docs_folder_reads() -> tuple[bool, list[str]]:
    """Try read_document on every supported file under docs/."""
    from src.document_readers import IngestReadOptions, read_document
    from src.local_retrieval import iter_supported_files

    docs = ROOT / "docs"
    if not docs.exists():
        return False, [f"Missing docs/ at {docs}"]

    options = IngestReadOptions.from_env()
    lines: list[str] = []
    ok = True
    found_suffixes: set[str] = set()

    for path in iter_supported_files(docs):
        rel = path.relative_to(docs)
        found_suffixes.add(path.suffix.lower())
        try:
            text = read_document(path, options)
            status = "OK" if text.strip() else "EMPTY"
            if status == "EMPTY":
                ok = False
            lines.append(f"  [{status}] {rel} ({len(text)} chars)")
        except Exception as exc:
            ok = False
            lines.append(f"  [FAIL] {rel} — {exc}")

    if not found_suffixes:
        lines.append("  (no supported files in docs/)")
        ok = False
    else:
        lines.insert(0, f"  Suffixes in docs/: {', '.join(sorted(found_suffixes))}")

    return ok, lines


def check_git_status() -> list[str]:
    result = subprocess.run(
        ["git", "status", "-sb"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ["  (not a git repo or git unavailable)"]
    return [f"  {line}" for line in result.stdout.strip().splitlines()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-push validation for MA-RAG")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip eval_kb and Ollama end-to-end (faster)",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Do not re-run python ingest.py on docs/",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures: list[str] = []

    _header("1. Environment")
    ollama_ok, ollama_msg = check_ollama()
    print(f"  Ollama: {ollama_msg}")
    if not ollama_ok and not args.quick:
        failures.append("Ollama not running (required for full check)")

    llm_ok, llm_msg = check_llm_config()
    print(f"  LLM: {llm_msg}")

    _header("2. Git status (informational)")
    for line in check_git_status():
        print(line)

    _header("3. docs/ file read smoke test")
    reads_ok, read_lines = check_docs_folder_reads()
    for line in read_lines:
        print(line)
    if not reads_ok:
        failures.append("One or more docs/ files failed to read")

    if not args.skip_ingest:
        _header("4. Rebuild index from docs/")
        code = _run([sys.executable, "ingest.py", "./docs"])
        if code != 0:
            failures.append("ingest.py failed")
    else:
        _header("4. Index check (ingest skipped)")
        index_ok, index_msg = check_docs_index()
        print(f"  {index_msg}")
        if not index_ok:
            failures.append(index_msg)

    index_ok, index_msg = check_docs_index()
    print(f"  {index_msg}")
    if not index_ok:
        failures.append(index_msg)

    _header("5. Automated tests (pytest)")
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT)}
    print("$ PYTHONPATH=. pytest -q")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        env=env,
    )
    if result.returncode != 0:
        failures.append("pytest failed")

    if args.quick:
        _header("6. eval_kb (skipped — --quick)")
        print("  Run without --quick before push to validate SLM + agents end-to-end.")
    else:
        if not ollama_ok:
            _header("6. eval_kb (skipped — Ollama down)")
            failures.append("eval_kb skipped because Ollama is down")
        else:
            _header("6. Knowledge-base eval (eval_kb.py)")
            env = {**__import__("os").environ, "PYTHONPATH": str(ROOT)}
            print("$ python eval_kb.py")
            result = subprocess.run(
                [sys.executable, "eval_kb.py"],
                cwd=ROOT,
                env=env,
            )
            if result.returncode != 0:
                failures.append("eval_kb.py failed (not 8/8)")

    _header("SUMMARY")
    if failures:
        print("FAILED — fix before push:")
        for item in failures:
            print(f"  ✗ {item}")
        print()
        print("Quick retry after fixes:")
        print("  python scripts/pre_push_check.py")
        return 1

    print("ALL CHECKS PASSED — safe to commit and push.")
    print()
    print("Suggested:")
    print("  git add -A")
    print('  git commit -m "your message"')
    print("  git push -u origin feature/track-b-agentic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
