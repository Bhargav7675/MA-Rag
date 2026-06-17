"""Centralized environment and path configuration for MA-RAG."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_openai_api_key() -> str:
    """Resolve OpenAI API key from OPENAI_API_KEY or legacy API_KEY."""
    key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    if not key:
        raise RuntimeError(
            "Missing OpenAI API key. Set OPENAI_API_KEY (or legacy API_KEY) in .env"
        )
    return key


def get_llm_provider() -> str:
    """LLM backend: openai (cloud API) or ollama (on-prem SLM)."""
    return _normalize_provider(os.getenv("MA_RAG_LLM_PROVIDER", "openai"))


def _normalize_provider(raw: str) -> str:
    value = raw.strip().lower()
    if value not in {"openai", "ollama"}:
        raise RuntimeError(
            f"Invalid LLM provider {raw!r}. Use 'openai' or 'ollama'."
        )
    return value


def _agent_env_key(agent_id: str) -> str:
    return agent_id.upper().replace("-", "_")


def get_agent_llm_provider(agent_id: str | None = None) -> str:
    """Per-agent provider override via MA_RAG_<AGENT>_PROVIDER (e.g. MA_RAG_PLANNER_PROVIDER)."""
    if agent_id:
        key = f"MA_RAG_{_agent_env_key(agent_id)}_PROVIDER"
        raw = os.getenv(key)
        if raw:
            return _normalize_provider(raw)
    return get_llm_provider()


def get_agent_ollama_model(agent_id: str | None = None) -> str:
    """Per-agent Ollama model override via MA_RAG_<AGENT>_OLLAMA_MODEL."""
    if agent_id:
        key = f"MA_RAG_{_agent_env_key(agent_id)}_OLLAMA_MODEL"
        raw = os.getenv(key)
        if raw:
            return raw.strip()
    return get_ollama_model()


def get_model_name() -> str:
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    if not model:
        raise RuntimeError("Missing MODEL_NAME in .env")
    return model


def get_ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "llama3.2:3b")


def get_ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def get_data_dir() -> Path:
    raw = os.getenv("MA_RAG_DATA_DIR", str(PROJECT_ROOT / "data"))
    return Path(raw).expanduser().resolve()


def get_evidence_ledger_dir() -> Path:
    raw = os.getenv(
        "MA_RAG_EVIDENCE_LEDGER_DIR",
        str(get_data_dir() / "evidence_ledger"),
    )
    return Path(raw).expanduser().resolve()


def get_index_dir() -> Path:
    raw = os.getenv("MA_RAG_INDEX_DIR", str(PROJECT_ROOT / "save_embs" / "gte-ml-base"))
    return Path(raw).expanduser().resolve()


def get_index_dataset_name() -> str:
    return os.getenv("MA_RAG_INDEX_DATASET", "dpr100")


def get_local_index_dir() -> Path:
    raw = os.getenv("MA_RAG_LOCAL_INDEX_DIR", str(PROJECT_ROOT / "local_index"))
    return Path(raw).expanduser().resolve()


def get_embedding_model_name() -> str:
    # Upstream default; override with MA_RAG_EMBEDDING_MODEL if needed.
    return os.getenv("MA_RAG_EMBEDDING_MODEL", "Alibaba-NLP/gte-multilingual-base")


def get_local_embedding_model_name() -> str:
    return os.getenv("MA_RAG_LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def get_local_embedding_backend() -> str:
    return os.getenv("MA_RAG_LOCAL_EMBEDDING_BACKEND", "hashing")


def get_api_host() -> str:
    return os.getenv("MA_RAG_API_HOST", "127.0.0.1")


def get_api_port() -> int:
    return int(os.getenv("MA_RAG_API_PORT", "8000"))


def get_pubmed_corpus_path() -> Path | None:
    raw = os.getenv("MA_RAG_PUBMED_CORPUS")
    if raw:
        return Path(raw).expanduser().resolve()
    candidate = get_data_dir() / "pubmed.tsv.gz"
    return candidate if candidate.exists() else None


BENCHMARK_DATASET_FILES = {
    "nq": "nq-dev-kilt.jsonl",
    "hotpotqa": "hotpotqa-dev-kilt.jsonl",
    "triviaqa": "triviaqa-dev-kilt.jsonl",
    "2wiki": "2WikiMultihopQA.jsonl",
    "fever": "fever-dev-kilt.jsonl",
}


def benchmark_dataset_path(name: str) -> Path:
    filename = BENCHMARK_DATASET_FILES.get(name)
    if not filename:
        raise ValueError(f"Unknown benchmark dataset: {name}")
    path = get_data_dir() / "benchmarks" / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {path}\n"
            "Download KILT dev JSONL files into data/benchmarks/ (see data/README.md)."
        )
    return path
