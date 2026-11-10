"""Shared pytest configuration.

Adds the ``--live`` flag, which enables the small number of tests that make a
real network request to Open-Meteo.

Those tests are **skipped by default** on purpose. A test suite that needs the
internet is not a test suite: it fails on a train, in a CI container without
egress, and in front of an audience. But a purely mocked suite can also pass
while the real API has changed its field names underneath it, so one live
smoke test is kept and run deliberately:

    python -m pytest                 # the whole suite, no network
    python -m pytest --live          # the same, plus the live tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _clear_forecast_cache():
    """Empty the weather module's forecast cache before every test.

    ``peakcast.weather`` caches whole API payloads in a module-level dict for
    ten minutes, which is exactly right in production - a conversation asks
    about the same place several times - and a trap in a test suite, because
    the cache outlives a test.

    It was found the hard way: the app test that monkeypatches ``requests.get``
    to fail and expects the offline badge **passed on its own and failed under
    ``--live``**. The live tests had already fetched Lienz, so the cache
    answered from the warm payload, ``requests.get`` was never called, and the
    badge truthfully said "live forecast". The test was right and the isolation
    was wrong.

    Autouse and in conftest rather than per-module, so no future test file can
    reintroduce the same leak by forgetting it.
    """
    from peakcast.weather import clear_cache

    clear_cache()
    yield
    clear_cache()


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
