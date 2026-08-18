"""Pytest configuration for the NanoCockpit project."""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "architecture: tests for the NANOCKPIT_ARCHITECTURE.md documentation file.",
    )
    config.addinivalue_line(
        "markers",
        "documentation: tests covering project documentation quality.",
    )


@pytest.fixture(scope="session")
def project_root():
    """Session-scoped fixture returning the absolute project root."""
    from pathlib import Path
    return Path(__file__).resolve().parent.parent
