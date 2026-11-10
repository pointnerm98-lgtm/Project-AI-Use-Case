"""Unit tests for the weather service.

The HTTP layer is mocked throughout, so the suite runs with no network. What
the mocks are used to pin down:

* the **failure paths**, which are the ones that matter in a live demo - a
  timeout, an HTTP error and a malformed response must all produce an offline
  report rather than an exception, and must say ``source="offline"``;
* the **part-of-day aggregation**, where hourly values are reduced with a
  different rule per variable (max for gusts, sum for precipitation, mean for
  cloud) and a wrong rule would be invisible in the output;
* the **sky mapping thresholds**, including the rule that precipitation always
  implies cloud, so the bot can never report "sunny" for a raining hour.

One live smoke test is kept at the bottom, skipped unless ``--live`` is
passed: a fully mocked suite can pass while the real API has renamed a field.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from peakcast.slots import (
    PART_AFTERNOON,
    PART_MORNING,
    PART_NIGHT,
    DaySlot,
    Place,
)
from peakcast.weather import (
    CLOUD_CLOUDY_FROM,
    CLOUD_SUNNY_BELOW,
    SKY_CLOUDY,
    SKY_FOGGY,
    SKY_PARTLY_CLOUDY,
    SKY_SUNNY,
    SOURCE_LIVE,
    SOURCE_OFFLINE,
    ForecastOutOfRangeError,
    clear_cache,
    describe_sky,
    get_forecast,
    load_offline_dataset,
    offline_forecast,
)

TODAY = date(2026, 11, 4)

INNSBRUCK = Place(
    name="Innsbruck",
    state="Tirol",
    lat=47.2692,
    lon=11.4041,
    elevation=570,
    population=132493,
    feature_code="PPLA",
)

GLOCKNER = Place(
    name="Großglockner",
    state="Tirol",
    lat=47.0745,
    lon=12.6939,
    elevation=3798,
    population=0,
    feature_code="MT",
)


def a_day(offset: int = 0, part: str | None = None) -> DaySlot:
    """Build a ``DaySlot`` for the tests.

    Args:
        offset: Days from ``TODAY``.
        part: Part-of-day flag.

    Returns:
        The day slot.
    """
    from datetime import timedelta

    return DaySlot(
        target_date=TODAY + timedelta(days=offset),
        offset_days=offset,
        part_of_day=part,
        in_range=0 <= offset <= 6,
    )


# The forecast cache is cleared before every test by an autouse fixture in
# conftest.py, so no per-module fixture is needed here. It lives there because
# the leak it prevents is not specific to this file: an app test that expects
# the offline fallback was once satisfied by a payload these tests had cached.


# --- Fake HTTP -------------------------------------------------------------


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        """Mimic requests' error behaviour for non-2xx codes."""
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        """Return the canned payload."""
        return self._payload


def daily_payload(
    *,
    days: int = 8,
    temperature_max: float = 9.4,
    temperature_min: float = 3.3,
    weather_code: int = 3,
    cloud_cover: float = 80.0,
    wind_speed: float = 12.0,
    wind_gusts: float = 30.0,
    rain: float = 1.2,
    snowfall: float = 0.0,
    probability: int = 40,
    start: date = TODAY,
) -> dict[str, Any]:
    """Build a payload shaped like the real daily response.

    Args:
        days: How many days to include.
        start: First date in the series.
        (the rest): The value repeated on every day.

    Returns:
        A payload dict.
    """
    from datetime import timedelta

    return {
        "elevation": 570.0,
        "daily": {
            "time": [(start + timedelta(days=i)).isoformat() for i in range(days)],
            "temperature_2m_max": [temperature_max] * days,
            "temperature_2m_min": [temperature_min] * days,
            "weather_code": [weather_code] * days,
            "cloud_cover_mean": [cloud_cover] * days,
            "wind_speed_10m_max": [wind_speed] * days,
            "wind_gusts_10m_max": [wind_gusts] * days,
            "rain_sum": [rain] * days,
            "snowfall_sum": [snowfall] * days,
            "precipitation_probability_max": [probability] * days,
        },
    }


def patch_requests(monkeypatch: pytest.MonkeyPatch, handler: Any) -> list[dict]:
    """Replace ``requests.get`` with ``handler`` and record the calls.

    Args:
        monkeypatch: pytest's monkeypatch fixture.
        handler: Called as ``handler(call_number)``; returns a response or
            raises.

    Returns:
        A list that receives one dict per call, for assertions on the
        parameters actually sent.
    """
    import requests

    calls: list[dict] = []

    def fake_get(url: str, params: dict | None = None, timeout: float | None = None):
        calls.append({"url": url, "params": params or {}, "timeout": timeout})
        return handler(len(calls))

    monkeypatch.setattr(requests, "get", fake_get)
    return calls


# --- The sky mapping -------------------------------------------------------


@pytest.mark.parametrize(
    "weather_code, cloud_cover, expected",
    [
        # fog wins over everything: a foggy summit has no view whatever the
        # cloud-cover figure says
        (45, 5.0, SKY_FOGGY),
        (48, 95.0, SKY_FOGGY),
        # clear and almost clear
        (0, 0.0, SKY_SUNNY),
        (1, 10.0, SKY_SUNNY),
        # the documented thresholds, checked exactly on the boundaries
        (1, CLOUD_SUNNY_BELOW - 0.1, SKY_SUNNY),
        (1, CLOUD_SUNNY_BELOW, SKY_PARTLY_CLOUDY),
        (2, 50.0, SKY_PARTLY_CLOUDY),
        (3, CLOUD_CLOUDY_FROM - 0.1, SKY_PARTLY_CLOUDY),
        (3, CLOUD_CLOUDY_FROM, SKY_CLOUDY),
        (3, 100.0, SKY_CLOUDY),
    ],
)
def test_sky_mapping(weather_code: int, cloud_cover: float, expected: str) -> None:
    """The documented thresholds must hold, boundaries included."""
    assert describe_sky(weather_code, cloud_cover) == expected


@pytest.mark.parametrize(
    "weather_code",
    [51, 61, 65, 71, 75, 80, 85, 95, 99],
)
def test_precipitation_always_implies_cloud(weather_code: int) -> None:
    """The bot must never call an hour "sunny" while it is raining on it.

    The cloud-cover thresholds alone would allow exactly that, because a
    shower can fall from a sky that averages 20 % cloud over the day.
    """
    assert describe_sky(weather_code, 0.0) == SKY_CLOUDY


# --- The live path ---------------------------------------------------------


def test_live_forecast_maps_the_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful request must produce a live report with mapped values."""
    patch_requests(monkeypatch, lambda n: FakeResponse(daily_payload()))

    report = get_forecast(INNSBRUCK, a_day(0))

    assert report.source == SOURCE_LIVE
    assert report.is_offline is False
    assert report.place_name == "Innsbruck"
    assert report.state == "Tirol"
    assert report.target_date == TODAY
    assert report.temperature_max == 9.4
    assert report.temperature_min == 3.3
    assert report.wind_gusts_max == 30.0
    assert report.rain_sum == 1.2
    assert report.precipitation_probability_max == 40
    assert report.sky == SKY_CLOUDY        # code 3, cloud 80 %
    assert report.model_elevation == 570.0


def test_the_places_elevation_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The elevation must go with the request.

    This is the decision that matters most for a mountain weather bot: without
    it Open-Meteo answers for the mean elevation of its grid cell, which near
    the Großglockner is ~150 m below the summit and about a degree warmer.
    """
    calls = patch_requests(monkeypatch, lambda n: FakeResponse(daily_payload()))

    get_forecast(GLOCKNER, a_day(0))

    assert calls[0]["params"]["elevation"] == 3798
    assert calls[0]["params"]["timezone"] == "Europe/Vienna"
    # Units are pinned explicitly so a change of API defaults cannot silently
    # turn km/h into m/s in the answers.
    assert calls[0]["params"]["wind_speed_unit"] == "kmh"
    assert calls[0]["params"]["temperature_unit"] == "celsius"


def test_hourly_is_not_requested_without_a_part_of_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A whole-day question must not pull 168 hours of data it will not use."""
    calls = patch_requests(monkeypatch, lambda n: FakeResponse(daily_payload()))

    get_forecast(INNSBRUCK, a_day(1))

    assert "hourly" not in calls[0]["params"]


# --- Robustness: every failure must land in the offline dataset ------------


def test_timeout_falls_back_to_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """A timeout must produce demo data, not an exception."""
    import requests

    def always_timeout(call_number: int):
        raise requests.exceptions.Timeout("too slow")

    calls = patch_requests(monkeypatch, always_timeout)

    report = get_forecast(INNSBRUCK, a_day(0))

    assert report.source == SOURCE_OFFLINE
    assert report.is_offline is True
    # One retry, so exactly two attempts - no more, because repeated attempts
    # only lengthen the wait before the fallback that was always coming.
    assert len(calls) == 2


def test_http_error_falls_back_to_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 500 from the API must fall back."""
    patch_requests(monkeypatch, lambda n: FakeResponse({}, status_code=500))

    report = get_forecast(INNSBRUCK, a_day(0))

    assert report.source == SOURCE_OFFLINE


def test_malformed_response_falls_back_to_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 200 response without a daily block must fall back.

    Checked separately from the HTTP errors because this is the failure that a
    status-code check alone would miss.
    """
    patch_requests(monkeypatch, lambda n: FakeResponse({"error": True}))

    report = get_forecast(INNSBRUCK, a_day(0))

    assert report.source == SOURCE_OFFLINE


def test_one_retry_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single dropped request must not cost the live answer."""
    import requests

    def fail_then_succeed(call_number: int):
        if call_number == 1:
            raise requests.exceptions.ConnectionError("dropped")
        return FakeResponse(daily_payload())

    calls = patch_requests(monkeypatch, fail_then_succeed)

    report = get_forecast(INNSBRUCK, a_day(0))

    assert report.source == SOURCE_LIVE
    assert len(calls) == 2


def test_cache_avoids_a_second_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """Asking about the same place twice must cost one request.

    This is the "and what about tomorrow?" case from the concept: a follow-up
    about a different day at the same place is answered from the cached
    payload, which already covers the whole week.
    """
    calls = patch_requests(monkeypatch, lambda n: FakeResponse(daily_payload()))

    first = get_forecast(INNSBRUCK, a_day(0))
    second = get_forecast(INNSBRUCK, a_day(2))

    assert first.source == SOURCE_LIVE
    assert second.source == SOURCE_LIVE
    assert second.target_date != first.target_date
    assert len(calls) == 1


def test_a_different_place_is_not_served_from_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cache must be keyed by location, not shared across places."""
    calls = patch_requests(monkeypatch, lambda n: FakeResponse(daily_payload()))

    get_forecast(INNSBRUCK, a_day(0))
    get_forecast(GLOCKNER, a_day(0))

    assert len(calls) == 2


# --- Part-of-day aggregation ----------------------------------------------


def hourly_payload(
    values_by_hour: dict[int, dict[str, float]],
    default_temperature: float = -99.0,
) -> dict[str, Any]:
    """Build a payload with hourly data for three days.

    Args:
        values_by_hour: Hour of day (0-23 of the first day) -> the values to
            put there. Hours not listed get ``default_temperature`` and zeros.
        default_temperature: Temperature for every hour not listed. The default
            of -99 is a sentinel: if it ever shows up in a result, the window
            picked up an hour it should not have. Tests whose window
            legitimately spans unlisted hours pass a neutral value instead.

    Returns:
        A payload with both a daily and an hourly block.
    """
    from datetime import timedelta

    payload = daily_payload(days=3)
    times: list[str] = []
    series: dict[str, list[float]] = {
        "temperature_2m": [],
        "weather_code": [],
        "cloud_cover": [],
        "wind_speed_10m": [],
        "wind_gusts_10m": [],
        "rain": [],
        "snowfall": [],
        "precipitation_probability": [],
    }

    for day_offset in range(3):
        for hour in range(24):
            stamp = TODAY + timedelta(days=day_offset)
            times.append(f"{stamp.isoformat()}T{hour:02d}:00")
            given = values_by_hour.get(hour, {}) if day_offset == 0 else {}
            for name in series:
                default = 0.0 if name != "temperature_2m" else default_temperature
                series[name].append(given.get(name, default))

    payload["hourly"] = {"time": times, **series}
    return payload


def test_part_of_day_aggregates_each_variable_with_its_own_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hourly values must be reduced per variable, not averaged uniformly.

    Afternoon is 12:00-18:00, so only those six hours may contribute. The
    temperature outside the window is set to -99 in the fixture, which would
    show up immediately if the window were wrong.
    """
    afternoon = {
        hour: {
            "temperature_2m": temp,
            "cloud_cover": cloud,
            "wind_speed_10m": 10.0,
            "wind_gusts_10m": gust,
            "rain": 0.5,
            "snowfall": 0.0,
            "precipitation_probability": prob,
            "weather_code": code,
        }
        for hour, temp, cloud, gust, prob, code in [
            (12, 5.0, 100.0, 20.0, 10, 3),
            (13, 7.0, 100.0, 40.0, 20, 3),
            (14, 9.0, 0.0, 61.0, 80, 61),   # the worst hour
            (15, 8.0, 0.0, 30.0, 30, 3),
            (16, 6.0, 100.0, 25.0, 10, 3),
            (17, 4.0, 0.0, 15.0, 5, 1),
        ]
    }
    patch_requests(monkeypatch, lambda n: FakeResponse(hourly_payload(afternoon)))

    report = get_forecast(INNSBRUCK, a_day(0, PART_AFTERNOON))

    assert report.source == SOURCE_LIVE
    assert report.part_of_day == PART_AFTERNOON
    assert report.temperature_max == 9.0          # max over the window
    assert report.temperature_min == 4.0          # min, and never -99
    assert report.wind_gusts_max == 61.0          # max gust, not the mean
    assert report.rain_sum == 3.0                 # 6 hours x 0.5 mm, summed
    assert report.precipitation_probability_max == 80
    assert report.cloud_cover == 50               # mean of 100,100,0,0,100,0
    assert report.weather_code == 61              # the worst code in the window
    assert report.sky == SKY_CLOUDY               # because it rains


def test_part_of_day_requests_hourly_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hourly variables must be asked for when a part of day is wanted."""
    calls = patch_requests(
        monkeypatch, lambda n: FakeResponse(hourly_payload({}))
    )

    get_forecast(INNSBRUCK, a_day(0, PART_MORNING))

    assert "hourly" in calls[0]["params"]
    assert "wind_gusts_10m" in calls[0]["params"]["hourly"]


def test_night_window_crosses_midnight(monkeypatch: pytest.MonkeyPatch) -> None:
    """"Tonight" must include the small hours of the next morning.

    Someone asking "how cold tonight" at 18:00 means the hours they will be
    out in, which run past midnight. The fixture puts a very cold hour at
    03:00 on the following day; a window that stopped at midnight would miss
    it and report a milder night than the user will actually get.
    """
    from datetime import timedelta

    # A neutral baseline, because the night window legitimately spans hours
    # this test does not set individually.
    payload = hourly_payload({}, default_temperature=0.0)
    hours = payload["hourly"]["time"]
    temps = payload["hourly"]["temperature_2m"]

    # The coldest hour of the night sits after midnight, on the NEXT day.
    temps[hours.index(f"{(TODAY + timedelta(days=1)).isoformat()}T03:00")] = -12.0
    # A milder hour before midnight, so min and max are distinguishable.
    temps[hours.index(f"{TODAY.isoformat()}T22:00")] = 2.0
    # A sentinel well outside the window: if the window leaked into the
    # afternoon, the maximum would be 40 instead of 2.
    temps[hours.index(f"{TODAY.isoformat()}T14:00")] = 40.0

    patch_requests(monkeypatch, lambda n: FakeResponse(payload))

    report = get_forecast(INNSBRUCK, a_day(0, PART_NIGHT))

    assert report.temperature_min == -12.0, "the window must cross midnight"
    assert report.temperature_max == 2.0, "the window must not reach the afternoon"


# --- The forecast horizon --------------------------------------------------


def test_out_of_range_raises_rather_than_serving_demo_data() -> None:
    """A day beyond the horizon must raise, not fall back.

    The distinction matters: the network being down is a reason to show demo
    data, but a question about 9 days out has an honest answer ("I can only
    forecast 7 days ahead"), and quietly inventing numbers instead would be
    worse than saying so.
    """
    too_far = DaySlot(
        target_date=date(2026, 11, 13), offset_days=9, in_range=False
    )
    with pytest.raises(ForecastOutOfRangeError):
        get_forecast(INNSBRUCK, too_far)


# --- The offline dataset ---------------------------------------------------


def test_offline_dataset_is_keyed_by_offset_not_by_date() -> None:
    """The demo data must not go stale.

    Every day series is indexed 0-6 relative to "today", so the dataset is
    still correct next year. A fixed date anywhere in the values would break
    that, so this test also checks none crept in.
    """
    import re

    dataset = load_offline_dataset()

    for place in dataset["places"].values():
        assert len(place["days"]) == 7
        for day in place["days"]:
            assert "time" not in day
            assert "date" not in day

    # And no ISO date string anywhere in the values, which is how a fixed date
    # would most likely creep back in.
    values_only = json.dumps(
        {"places": dataset["places"], "profiles": dataset["elevation_profiles"]}
    )
    assert not re.search(r"\d{4}-\d{2}-\d{2}", values_only)


def test_offline_dataset_covers_eight_places_towns_and_peaks() -> None:
    """The demo set must contain both valley towns and high summits."""
    dataset = load_offline_dataset()
    places = dataset["places"]

    assert len(places) == 8
    elevations = [p["elevation"] for p in places.values()]
    assert sum(1 for e in elevations if e < 1000) >= 3     # towns
    assert sum(1 for e in elevations if e > 1900) >= 3     # peaks


def test_offline_values_are_physically_consistent() -> None:
    """Fictional does not mean careless.

    An earlier draft of the dataset reported rain at -5.8 °C on the
    Schmittenhöhe, because the snow line was a fixed per-day constant instead
    of following the temperature. Implausible demo data undermines the demo,
    so the property is pinned here.
    """
    dataset = load_offline_dataset()
    series = [p["days"] for p in dataset["places"].values()]
    series += [p["days"] for p in dataset["elevation_profiles"]]

    for days in series:
        for day in days:
            if day["rain_sum"] > 0:
                assert day["temperature_2m_max"] > 1.0, day
            if day["snowfall_sum"] > 0:
                assert day["temperature_2m_max"] <= 4.0, day
            assert day["temperature_2m_min"] <= day["temperature_2m_max"]
            assert day["wind_gusts_10m_max"] >= day["wind_speed_10m_max"]
            assert 0 <= day["cloud_cover_mean"] <= 100
            assert 0 <= day["precipitation_probability_max"] <= 100


@pytest.mark.parametrize("offset", range(7))
def test_offline_forecast_covers_every_day_in_range(offset: int) -> None:
    """Each of the seven days must produce a report."""
    report = offline_forecast(GLOCKNER, a_day(offset))

    assert report.source == SOURCE_OFFLINE
    assert report.place_name == "Großglockner"
    # A 3 798 m summit is below freezing on every day of this dataset's story.
    assert report.temperature_max < 0


def test_unlisted_place_uses_an_elevation_banded_profile() -> None:
    """A place not in the demo set must still get a plausible answer.

    Elevation decides whether a plausible answer is +12 °C and rain or -6 °C
    and snow, so the generic profiles are banded by elevation rather than
    being one flat default - otherwise offline mode would answer a question
    about a 3 000 m summit with valley numbers.
    """
    valley = Place(
        name="Nowhere Village", state="Tirol", lat=47.0, lon=11.0,
        elevation=600, population=100, feature_code="PPL",
    )
    summit = Place(
        name="Nowhere Peak", state="Tirol", lat=47.0, lon=11.0,
        elevation=2800, population=0, feature_code="PK",
    )

    valley_report = offline_forecast(valley, a_day(1))
    summit_report = offline_forecast(summit, a_day(1))

    assert valley_report.temperature_max > summit_report.temperature_max + 8


def test_offline_part_of_day_is_adjusted() -> None:
    """A part-of-day question must not return the untouched daily figures."""
    whole_day = offline_forecast(INNSBRUCK, a_day(1))
    morning = offline_forecast(INNSBRUCK, a_day(1, PART_MORNING))

    assert morning.part_of_day == PART_MORNING
    # Mornings are colder than the daily maximum and less windy.
    assert morning.temperature_max < whole_day.temperature_max
    assert morning.wind_gusts_max < whole_day.wind_gusts_max


def test_offline_mode_can_be_forced() -> None:
    """``allow_live=False`` must skip the network entirely.

    Used by the demo mode, and by any test that wants the offline path without
    having to mock HTTP.
    """
    report = get_forecast(INNSBRUCK, a_day(0), allow_live=False)
    assert report.source == SOURCE_OFFLINE


# --- Live smoke test (skipped unless --live) -------------------------------


def a_real_day(offset: int = 0, part: str | None = None) -> DaySlot:
    """Build a day slot anchored to the ACTUAL current date in Vienna.

    The mocked tests use the fixed ``TODAY`` constant, which is what keeps
    them reproducible - the fake payloads are built around the same constant,
    so the two always agree.

    The live tests cannot do that, and getting this wrong is how they broke:
    they asked the real API for the fixed date, which worked on the day it was
    written and started failing the next morning, because a forecast service
    does not serve yesterday. A live test has to use the real clock.

    Args:
        offset: Days from today.
        part: Part-of-day flag.

    Returns:
        The day slot.
    """
    from datetime import timedelta

    from peakcast.slots import today_in_vienna

    today = today_in_vienna()
    return DaySlot(
        target_date=today + timedelta(days=offset),
        offset_days=offset,
        part_of_day=part,
        in_range=True,
    )


@pytest.mark.live
def test_live_api_still_returns_what_we_parse() -> None:
    """Fetch two real forecasts and check the parsing still holds.

    The point of this test is the one thing mocks cannot catch: the real API
    renaming or dropping a field. It checks a valley town and a 3 798 m summit,
    because the elevation parameter is the part most specific to this project.
    """
    from peakcast.slots import today_in_vienna

    for place in (INNSBRUCK, GLOCKNER):
        clear_cache()
        day = a_real_day(0)
        report = get_forecast(place, day)

        assert report.source == SOURCE_LIVE, "the live API should have answered"
        assert report.place_name == place.name
        assert report.target_date == today_in_vienna()
        assert -40.0 < report.temperature_min <= report.temperature_max < 45.0
        assert report.wind_gusts_max >= report.wind_speed_max >= 0
        assert 0 <= report.cloud_cover <= 100
        assert report.sky in {
            SKY_SUNNY, SKY_PARTLY_CLOUDY, SKY_CLOUDY, SKY_FOGGY
        }
        # The elevation we asked for must be the one the model used.
        assert report.model_elevation is not None
        assert abs(report.model_elevation - (place.elevation or 0)) < 1.0


@pytest.mark.live
def test_live_part_of_day_aggregation() -> None:
    """The hourly path must work against the real API too."""
    clear_cache()
    report = get_forecast(INNSBRUCK, a_real_day(1, PART_AFTERNOON))

    assert report.source == SOURCE_LIVE
    assert report.part_of_day == PART_AFTERNOON
    assert report.temperature_min <= report.temperature_max
