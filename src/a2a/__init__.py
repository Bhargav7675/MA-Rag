"""Agent-to-agent scaffolding (in-process bus + registry)."""

from src.a2a.bus import InProcessA2ABus
from src.a2a.envelope import A2AEnvelope, A2AResponse
from src.a2a.file_journal import A2AFileJournal
from src.a2a.registry import AgentDescriptor, AgentRegistry

__all__ = [
    "A2AEnvelope",
    "A2AResponse",
    "A2AFileJournal",
    "AgentDescriptor",
    "AgentRegistry",
    "InProcessA2ABus",
]
