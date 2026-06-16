"""Append-only evidence ledger (JSONL per run) for workflow audit."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.env import get_evidence_ledger_dir


class EvidenceLedger:
    def __init__(self, run_id: str, *, base_dir: Optional[Path] = None):
        self.run_id = run_id
        self.base_dir = base_dir or get_evidence_ledger_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.base_dir / f"{run_id}.jsonl"

    def append(
        self,
        *,
        agent: str,
        workflow_step: str,
        payload: dict[str, Any],
    ) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "agent": agent,
            "workflow_step": workflow_step,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
