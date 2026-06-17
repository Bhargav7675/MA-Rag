"""A2A file journal tests."""

from pathlib import Path

from src.a2a.bus import InProcessA2ABus
from src.a2a.file_journal import A2AFileJournal
from src.a2a.registry import AgentDescriptor, AgentRegistry


def test_file_journal_records_bus_messages(tmp_path: Path):
    journal = A2AFileJournal(base_dir=tmp_path)
    registry = AgentRegistry()
    registry.register(
        AgentDescriptor(agent_id="echo", role="echo", message_types=["echo"]),
        lambda payload: {"echo": payload.get("text", "")},
    )
    bus = InProcessA2ABus(registry=registry, journal=journal)

    response = bus.request(
        from_agent="workflow",
        to_agent="echo",
        message_type="echo",
        payload={"text": "ping"},
        correlation_id="run123",
    )

    assert response.success is True
    journal_path = journal.path_for("run123")
    assert journal_path.exists()
    lines = journal_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
