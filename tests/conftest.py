"""Shared pytest configuration.

Adds the ``--live`` flag, which enables the small number of tests that make a
real network request to Open-Meteo.

Those tests are **skipped by default** on purpose. A test suite that needs the
internet is not a test suite: it fails on a train, in a CI container without
egress, and in front of an audience. But a purely mocked suite can also pass
while the real API has changed its field names underneath it, so one live
smoke test is kept and run deliberately:

    python -m pytest                 # 133+ tests, no network
    python -m pytest --live          # the same, plus the live smoke test
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--live`` command-line flag."""
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="also run tests that call the real Open-Meteo API",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``live`` marker so strict-marker mode stays usable."""
    config.addinivalue_line(
        "markers", "live: needs network access to the Open-Meteo API"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip every ``live``-marked test unless ``--live`` was passed."""
    if config.getoption("--live"):
        return
    skip_live = pytest.mark.skip(reason="needs --live to call the real API")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
