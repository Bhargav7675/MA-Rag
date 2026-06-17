"""Small local-document FAISS retrieval for Phase 0 development."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import faiss
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from src.env import (
    get_local_embedding_backend,
    get_local_embedding_model_name,
    get_local_index_dir,
)


INDEX_FILE = "index.faiss"
METADATA_FILE = "chunks.jsonl"


@dataclass
class LocalChunk:
    doc_id: str
    source: str
    chunk_index: int
    text: str


class HashingEmbedder:
    """Stateless CPU embedder for local smoke tests and small document sets."""

    def __init__(self, n_features: int = 1024):
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            stop_words="english",
        )

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        return self.vectorizer.transform(texts).toarray().astype("float32")


def local_index_exists(index_dir: Path | None = None) -> bool:
    index_dir = index_dir or get_local_index_dir()
    return (index_dir / INDEX_FILE).exists() and (index_dir / METADATA_FILE).exists()


def load_embedding_model(device: str | None = None):
    backend = get_local_embedding_backend().lower()
    if backend == "hashing":
        return HashingEmbedder()
    if backend not in {"sentence-transformers", "sentence_transformers"}:
        raise ValueError(
            "Unsupported MA_RAG_LOCAL_EMBEDDING_BACKEND. "
            "Use 'hashing' or 'sentence-transformers'."
        )

    from sentence_transformers import SentenceTransformer

    model_name = get_local_embedding_model_name()
    # CPU is the safest default on macOS; MPS has caused native segfaults with
    # some transformer/FAISS combinations.
    model = SentenceTransformer(model_name, device=device or "cpu")
    return model


def embed_texts(
    texts: list[str],
    model,
    *,
    batch_size: int = 16,
) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")


def read_supported_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "PDF ingestion requires pypdf. Run: pip install -r requirements.txt"
            ) from exc

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    raise ValueError(f"Unsupported file type: {path}")


def _is_excluded(path: Path, exclude_dir_names: Sequence[str]) -> bool:
    if not exclude_dir_names:
        return False
    return any(part in exclude_dir_names for part in path.parts)


def iter_supported_files(
    root: Path,
    *,
    exclude_dir_names: Sequence[str] = (),
) -> Iterable[Path]:
    supported = {".pdf", ".txt", ".md"}
    if root.is_file():
        if root.suffix.lower() in supported and not _is_excluded(root, exclude_dir_names):
            yield root
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in supported:
            continue
        if _is_excluded(path, exclude_dir_names):
            continue
        yield path


def chunk_text(text: str, *, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunks.append(normalized[start:end])
        if end == len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_local_index(
    input_path: Path,
    *,
    index_dir: Optional[Path] = None,
    chunk_size: int = 1200,
    overlap: int = 200,
    batch_size: int = 16,
    exclude_dir_names: Sequence[str] = ("wiki",),
) -> int:
    index_dir = index_dir or get_local_index_dir()
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    chunks: list[LocalChunk] = []
    for file_path in iter_supported_files(input_path, exclude_dir_names=exclude_dir_names):
        text = read_supported_file(file_path)
        rel_source = str(file_path.relative_to(input_path)) if input_path.is_dir() else file_path.name
        for chunk_index, chunk in enumerate(chunk_text(text, chunk_size=chunk_size, overlap=overlap)):
            chunks.append(
                LocalChunk(
                    doc_id=f"{rel_source}#{chunk_index}",
                    source=rel_source,
                    chunk_index=chunk_index,
                    text=chunk,
                )
            )

    if not chunks:
        raise RuntimeError(
            f"No supported content found under {input_path}. Add .pdf, .txt, or .md files."
        )

    model = load_embedding_model()
    embeddings = embed_texts(
        [chunk.text for chunk in chunks],
        model,
        batch_size=batch_size,
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / INDEX_FILE))
    with (index_dir / METADATA_FILE).open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    return len(chunks)


class LocalRetrieverTool:
    def __init__(self, *, index_dir: Path | None = None, top_k: int = 5):
        self.index_dir = index_dir or get_local_index_dir()
        if not local_index_exists(self.index_dir):
            raise FileNotFoundError(
                f"No local index found under {self.index_dir}. Run: python ingest.py ./docs"
            )
        self.index = faiss.read_index(str(self.index_dir / INDEX_FILE))
        self.chunks = self._load_chunks()
        self.top_k = top_k
        self.model = load_embedding_model()

    def _load_chunks(self) -> list[LocalChunk]:
        chunks: list[LocalChunk] = []
        with (self.index_dir / METADATA_FILE).open("r", encoding="utf-8") as f:
            for line in f:
                chunks.append(LocalChunk(**json.loads(line)))
        return chunks

    def __call__(self, query: str):
        hits = self.search_with_scores(query, top_k=self.top_k)
        docs = [text for _, text, _ in hits]
        doc_ids = [doc_id for doc_id, _, _ in hits]
        return docs, doc_ids

    def search_with_scores(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[tuple[str, str, float]]:
        k = top_k or self.top_k
        query_embedding = embed_texts([query], self.model)
        scores, indices = self.index.search(query_embedding, k)
        hits: list[tuple[str, str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[int(idx)]
            hits.append((chunk.doc_id, chunk.text, float(score)))
        return hits
