"""HTTP API package for MA-RAG."""

from src.api.server import fastapi_app

app = fastapi_app

__all__ = ["app", "fastapi_app"]
