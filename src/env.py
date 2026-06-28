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


def get_a2a_journal_dir() -> Path:
    raw = os.getenv(
        "MA_RAG_A2A_JOURNAL_DIR",
        str(get_data_dir() / "a2a_journal"),
    )
    return Path(raw).expanduser().resolve()


def get_a2a_queue_dir() -> Path:
    raw = os.getenv(
        "MA_RAG_A2A_QUEUE_DIR",
        str(get_data_dir() / "a2a_queue"),
    )
    return Path(raw).expanduser().resolve()


def get_a2a_transport() -> str:
    """A2A transport: in_process (default) or file_queue (remote workers)."""
    value = os.getenv("MA_RAG_A2A_TRANSPORT", "in_process").strip().lower()
    if value not in {"in_process", "file_queue"}:
        raise RuntimeError(
            f"Invalid A2A transport {value!r}. Use 'in_process' or 'file_queue'."
        )
    return value


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


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def get_fast_mode() -> bool:
    """Fewer LLM calls: skip planner/evidence/extract when heuristics suffice."""
    return _env_flag("MA_RAG_FAST_MODE")


def get_skip_rag_extract() -> bool:
    """Skip per-chunk extract LLM calls (use raw passages). Defaults on in fast mode."""
    if os.getenv("MA_RAG_SKIP_RAG_EXTRACT") is not None:
        return _env_flag("MA_RAG_SKIP_RAG_EXTRACT")
    return get_fast_mode()


def get_retrieval_top_k() -> int:
    raw = os.getenv("MA_RAG_RETRIEVAL_TOP_K")
    if raw:
        return max(1, int(raw))
    return 2 if get_fast_mode() else 3


def get_ollama_keep_alive() -> str:
    """How long Ollama keeps the model loaded between requests (e.g. 5m, -1)."""
    return os.getenv("OLLAMA_KEEP_ALIVE", "5m")


def get_api_host() -> str:
    return os.getenv("MA_RAG_API_HOST", "127.0.0.1")


def get_api_port() -> int:
    return int(os.getenv("MA_RAG_API_PORT", "8000"))


def get_pdf_password() -> str | None:
    """Default password for encrypted PDFs during ingest."""
    raw = os.getenv("MA_RAG_PDF_PASSWORD")
    return raw.strip() if raw else None


def get_pdf_passwords_file() -> Path | None:
    """JSON map of PDF filename (or \"*\") -> password."""
    raw = os.getenv("MA_RAG_PDF_PASSWORDS_FILE")
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def get_pdf_ocr_enabled() -> bool:
    """OCR scanned/image PDFs when embedded text is sparse."""
    return _env_flag("MA_RAG_PDF_OCR", default=True)


def get_pdf_ocr_min_chars_per_page() -> int:
    raw = os.getenv("MA_RAG_PDF_OCR_MIN_CHARS_PER_PAGE", "50")
    return max(1, int(raw))


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
