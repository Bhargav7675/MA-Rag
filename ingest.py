#!/usr/bin/env python3
"""Build a local FAISS index from corporate documents (PDF, Office, text)."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.document_readers import IngestReadOptions, SUPPORTED_SUFFIXES
from src.env import get_local_index_dir
from src.local_retrieval import (
    DEFAULT_INGEST_EXCLUDE_DIRS,
    DEFAULT_INGEST_EXCLUDE_FILES,
    build_local_index,
)

load_dotenv()


def parse_args():
    formats = ", ".join(sorted(SUPPORTED_SUFFIXES))
    parser = argparse.ArgumentParser(
        description=(
            "Ingest local documents for MA-RAG (default: ./docs). "
            f"Supported: {formats}"
        ),
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="./docs",
        help=f"File or directory with supported documents (default: ./docs)",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=get_local_index_dir(),
        help="Where to write the local FAISS index",
    )
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--include-wiki",
        action="store_true",
        help="Include docs/wiki/ in the index (excluded by default — demo pages rank above project KB)",
    )
    parser.add_argument(
        "--pdf-password",
        default=None,
        help="Password for encrypted PDFs (overrides MA_RAG_PDF_PASSWORD for this run)",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR for scanned/image-only PDFs",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    exclude_dir_names = () if args.include_wiki else DEFAULT_INGEST_EXCLUDE_DIRS
    exclude_file_names = DEFAULT_INGEST_EXCLUDE_FILES
    read_options = IngestReadOptions.from_env(
        pdf_password=args.pdf_password,
        enable_pdf_ocr=not args.no_ocr,
    )
    count = build_local_index(
        Path(args.path),
        index_dir=args.index_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        exclude_dir_names=exclude_dir_names,
        exclude_file_names=exclude_file_names,
        read_options=read_options,
    )
    excluded: list[str] = []
    if exclude_dir_names:
        excluded.append(f"dirs: {', '.join(exclude_dir_names)}")
    if exclude_file_names:
        excluded.append(f"files: {', '.join(exclude_file_names)}")
    if excluded:
        print(f"Excluded from index — {'; '.join(excluded)} (use --include-wiki to index wiki)")
    print(f"Indexed {count} chunks into {args.index_dir}")
    print()
    print("Next step — start the chat (questions go inside this program):")
    print("  python ask.py")
    print()
    print("At the Question> prompt, type your question. Type quit to exit.")
    print('One-shot example: python ask.py "What did they launch in 2021?"')


if __name__ == "__main__":
    main()
