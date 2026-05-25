#!/usr/bin/env python3
"""Build a small local FAISS index from PDFs, Markdown, or text files."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.env import get_local_index_dir
from src.local_retrieval import build_local_index

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest local documents for MA-RAG")
    parser.add_argument("path", help="File or directory containing .pdf, .txt, or .md files")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=get_local_index_dir(),
        help="Where to write the local FAISS index",
    )
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    count = build_local_index(
        Path(args.path),
        index_dir=args.index_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )
    print(f"Indexed {count} chunks into {args.index_dir}")
    print("Now ask with:")
    print('  python ask.py "your question"')


if __name__ == "__main__":
    main()
