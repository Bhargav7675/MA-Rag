#!/usr/bin/env python3
"""Run the MA-RAG FastAPI server."""

from src.runtime_warnings import configure_runtime_warnings

configure_runtime_warnings()

from src.api.server import main

if __name__ == "__main__":
    main()
