"""Agent registry for A2A routing (in-process; future remote agents)."""

from __future__ import annotations

from typing import Callable, Optional

from pydantic import BaseModel, Field


AgentHandler = Callable[[dict], dict]


class AgentDescriptor(BaseModel):
    agent_id: str
    role: str
    description: str = ""
    message_types: list[str] = Field(default_factory=list)
    version: str = "1.0.0"


class AgentRegistry:
    def __init__(self):
        self._descriptors: dict[str, AgentDescriptor] = {}
        self._handlers: dict[str, AgentHandler] = {}

    def register(
        self,
        descriptor: AgentDescriptor,
        handler: AgentHandler,
    ) -> None:
        self._descriptors[descriptor.agent_id] = descriptor
        self._handlers[descriptor.agent_id] = handler

    def get(self, agent_id: str) -> Optional[AgentDescriptor]:
        return self._descriptors.get(agent_id)

    def list_agents(self) -> list[AgentDescriptor]:
        return list(self._descriptors.values())

    def dispatch(self, agent_id: str, payload: dict) -> dict:
        handler = self._handlers.get(agent_id)
        if handler is None:
            raise KeyError(f"No handler registered for agent: {agent_id}")
        return handler(payload)
