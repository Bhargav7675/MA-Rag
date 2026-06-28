"""Shared LLM client helpers (OpenAI API or on-prem Ollama SLM)."""

from __future__ import annotations

import os
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.env import (
    get_agent_llm_provider,
    get_agent_ollama_model,
    get_llm_provider,
    get_model_name,
    get_ollama_base_url,
    get_ollama_model,
    get_openai_api_key,
)

_AGENT_IDS = (
    "planner",
    "rag_step",
    "evidence_curator",
    "step_definer",
    "summarizer",
    "critic",
)


def is_agent_ollama(agent_id: Optional[str] = None) -> bool:
    """True when the effective provider for this agent (or global default) is Ollama."""
    if agent_id:
        return get_agent_llm_provider(agent_id) == "ollama"
    return get_llm_provider() == "ollama"


def create_chat_llm(
    *,
    agent_id: Optional[str] = None,
    temperature: float = 0.0,
    max_retries: int = 5,
) -> BaseChatModel:
    provider = get_agent_llm_provider(agent_id)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        from src.env import get_ollama_keep_alive

        return ChatOllama(
            model=get_agent_ollama_model(agent_id),
            base_url=get_ollama_base_url(),
            temperature=temperature,
            keep_alive=get_ollama_keep_alive(),
        )

    return ChatOpenAI(
        model=get_model_name(),
        temperature=temperature,
        api_key=get_openai_api_key(),
        max_retries=max_retries,
    )


def describe_active_llm() -> str:
    default_provider = get_llm_provider()
    if default_provider == "ollama":
        base = f"ollama/{get_ollama_model()} @ {get_ollama_base_url()}"
    else:
        base = f"openai/{get_model_name()}"

    overrides: list[str] = []
    for agent in _AGENT_IDS:
        env_key = f"MA_RAG_{agent.upper()}_PROVIDER"
        if os.getenv(env_key):
            p = get_agent_llm_provider(agent)
            if p == "ollama":
                overrides.append(f"{agent}=ollama/{get_agent_ollama_model(agent)}")
            else:
                overrides.append(f"{agent}=openai/{get_model_name()}")
    if overrides:
        return f"{base} | per-agent: {', '.join(overrides)}"
    return base
