"""Append-only A2A message journal (queue-ready transport scaffolding)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.env import get_a2a_journal_dir


class A2AFileJournal:
    """
    Persists A2A envelopes and responses per workflow run_id.

    Future remote workers can tail these JSONL files instead of in-process dispatch.
    """

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or get_a2a_journal_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, correlation_id: str) -> Path:
        return self.base_dir / f"{correlation_id}.jsonl"

    def append(self, correlation_id: str, record: dict[str, Any]) -> None:
        path = self.path_for(correlation_id)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
