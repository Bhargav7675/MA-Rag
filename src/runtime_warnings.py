"""Suppress noisy third-party warnings on CLI and API startup."""

from __future__ import annotations

import warnings


def configure_runtime_warnings() -> None:
    # Message filters first — importing urllib3.exceptions triggers urllib3.__init__.
    warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")
    warnings.filterwarnings("ignore", message=".*allowed_objects.*")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic\.main")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"importlib\._bootstrap")
    warnings.filterwarnings("ignore", module=r"langgraph\..*")
