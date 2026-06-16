"""Shared LLM client helpers (OpenAI API or on-prem Ollama SLM)."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.env import (
    get_llm_provider,
    get_model_name,
    get_ollama_base_url,
    get_ollama_model,
    get_openai_api_key,
)


def create_chat_llm(*, temperature: float = 0.0, max_retries: int = 5) -> BaseChatModel:
    provider = get_llm_provider()
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=get_ollama_model(),
            base_url=get_ollama_base_url(),
            temperature=temperature,
        )

    return ChatOpenAI(
        model=get_model_name(),
        temperature=temperature,
        api_key=get_openai_api_key(),
        max_retries=max_retries,
    )


def describe_active_llm() -> str:
    provider = get_llm_provider()
    if provider == "ollama":
        return f"ollama/{get_ollama_model()} @ {get_ollama_base_url()}"
    return f"openai/{get_model_name()}"
