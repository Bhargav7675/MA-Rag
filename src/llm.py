"""Shared LangChain OpenAI client helpers."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.env import get_model_name, get_openai_api_key


def create_chat_llm(*, temperature: float = 0.0, max_retries: int = 5) -> ChatOpenAI:
    return ChatOpenAI(
        model=get_model_name(),
        temperature=temperature,
        api_key=get_openai_api_key(),
        max_retries=max_retries,
    )
