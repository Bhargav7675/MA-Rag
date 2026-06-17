"""Shared pytest fixtures."""

from __future__ import annotations

from src.runtime_warnings import configure_runtime_warnings

configure_runtime_warnings()

import pytest

from src.workflow.service import get_workflow_service


@pytest.fixture(autouse=True)
def _clear_workflow_service_cache():
    get_workflow_service.cache_clear()
    yield
    get_workflow_service.cache_clear()
