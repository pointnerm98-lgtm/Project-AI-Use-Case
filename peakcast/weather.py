"""Weather service for PeakCast: Open-Meteo client with an offline fallback.

Position in the architecture: the NLU has already decided *what* was asked
(`intent_classifier`), *where* and *when* (`slots`). This module turns a place
and a day into actual numbers, which the response generator then puts into a
sentence.

Design decisions
----------------

* **Open-Meteo, no API key.** Chosen in the concept because it needs no account
  and no licence key, so the project stays runnable from the submitted files.
  The free endpoint is for non-commercial use (noted in the README).

* **The place's elevation is sent with every request.** This is the decision
  that matters most for a *mountain* weather bot. Open-Meteo otherwise answers
  for the average elevation of its grid cell, which near a summit can be
  hundreds of metres off:

      Großglockner, no elevation parameter -> model elevation 3 650 m, -8.9 °C
      Großglockner, elevation=3798         -> model elevation 3 798 m, -9.8 °C

  A degree of temperature is the difference between rain and snow, so the
  parameter is not cosmetic. It is also why `scripts/build_places.py` keeps the
  elevation of every gazetteer row and why administrative divisions were
  excluded there - their elevation describes an area, not a place.

* **Every failure falls back to the offline dataset, and says so.** Any network
  problem returns a report with ``source="offline"`` rather than an exception,
  so a live demo cannot break in front of an audience. The report carries the
  source explicitly because a fictional forecast must never be presentable as
  a real one - the response generator is expected to say when sample data was
  used.

* **Part of day is computed from hourly data.** When the user said "this
  afternoon", averaging the whole day would answer a different question.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from peakcast.slots import (
    DATA_DIR,
    MAX_DAY_OFFSET,
    PART_AFTERNOON,
    PART_EVENING,
    PART_MORNING,
    PART_NIGHT,
    DaySlot,
    Place,
)

# --- Endpoint and request parameters --------------------------------------
API_URL: str = "https://api.open-meteo.com/v1/forecast"

# Units are requested explicitly rather than relying on the API defaults, so a
# change on their side cannot silently turn km/h into m/s in the answers.
# Note that Open-Meteo returns snowfall in **cm** and rain in **mm**, which is
# exactly the pair of units a ski tourer wants.
REQUEST_UNITS: dict[str, str] = {
    "temperature_unit": "celsius",
    "wind_speed_unit": "kmh",
    "precipitation_unit": "mm",
}

DAILY_VARIABLES: tuple[str, ...] = (
    "temperature_2m_max",
    "temperature_2m_min",
    "weather_code",
    "cloud_cover_mean",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",       # gusts decide whether a paraglider launches
    "rain_sum",
    "snowfall_sum",
    "precipitation_probability_max",
)

# The hourly equivalents, needed only when a part of day was extracted.
HOURLY_VARIABLES: tuple[str, ...] = (
    "temperature_2m",
    "weather_code",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "rain",
    "snowfall",
    "precipitation_probability",
)

# --- Robustness ------------------------------------------------------------
# A user standing at a trailhead on mobile data will not wait. Six seconds is
# long enough for a normal response and short enough that the offline fallback
# feels like an answer rather than a hang.
REQUEST_TIMEOUT_SECONDS: float = 6.0

# One retry, not more: the common failure is a single dropped packet, and
# repeated attempts only multiply the wait before the fallback that was always
# going to be used.
MAX_ATTEMPTS: int = 2

# Forecasts do not change faster than this, and a conversation asks about the
# same place several times in a row ("and what about tomorrow?"). Caching the
# whole API payload per location means those follow-ups cost no request at all.
CACHE_TTL_SECONDS: float = 600.0

OFFLINE_JSON: Path = DATA_DIR / "offline_forecasts.json"

SOURCE_LIVE: str = "live"
SOURCE_OFFLINE: str = "offline"

# --- Sky categories --------------------------------------------------------
SKY_SUNNY: str = "sunny"
SKY_PARTLY_CLOUDY: str = "partly cloudy"
SKY_CLOUDY: str = "cloudy"
SKY_FOGGY: str = "foggy"

# WMO weather codes for fog and depositing rime fog. Fog is kept as its own
# category rather than being folded into "cloudy" because for the use case it
# is a different answer: fog on a summit means no view and poor orientation,
# which is the thing `ask_sky` is really about.
FOG_CODES: frozenset[int] = frozenset({45, 48})

# The lowest WMO code that means precipitation (drizzle). Everything from here
# up implies an overcast sky, so the mapping never reports "sunny" for an hour
# that is also raining - an inconsistency the cloud-cover thresholds alone
# would allow.
FIRST_PRECIPITATION_CODE: int = 51

# Cloud-cover thresholds, in percent. These follow the meteorological okta
# convention: the sky is reported in eighths, "clear" is up to 2/8 (25 %) and
# "overcast" is 7/8 or more. 75 % is used for the upper bound rather than
# 87.5 % because the value here is a *mean* over the day or the window, and a
# daily mean of 75 % already means there was very little sun.
CLOUD_SUNNY_BELOW: float = 25.0
CLOUD_CLOUDY_FROM: float = 75.0

# --- Part-of-day windows ---------------------------------------------------
# Local Vienna hours, as [start, end) pairs. "night" deliberately crosses
# midnight: someone asking "how cold tonight" at 18:00 means the hours they
# will actually be out in, which run into the small hours of the next day.
PART_OF_DAY_HOURS: dict[str, tuple[int, int]] = {
    PART_MORNING: (6, 12),
    PART_AFTERNOON: (12, 18),
    PART_EVENING: (18, 23),
    PART_NIGHT: (21, 30),   # 21:00 today to 06:00 tomorrow
}


@dataclass(frozen=True)
class WeatherReport:
    """The forecast for one place on one day (or one part of a day).

    Attributes:
        place_name: Name of the place the forecast is for.
        state: Federal state, or ``""`` for features that span states.
        elevation: Metres above sea level that the forecast was requested for.
        target_date: The day this report describes.
        part_of_day: ``"morning"``, ``"afternoon"``, ``"evening"``, ``"night"``
            or ``None`` for the whole day.
        temperature_max: Highest temperature in °C.
        temperature_min: Lowest temperature in °C.
        weather_code: WMO code; the worst one in the window.
        cloud_cover: Mean cloud cover in percent.
        wind_speed_max: Highest mean wind speed in km/h.
        wind_gusts_max: Highest gust in km/h.
        rain_sum: Rainfall in mm.
        snowfall_sum: Snowfall in cm.
        precipitation_probability_max: Highest probability in percent.
        sky: One of ``sunny``, ``partly cloudy``, ``cloudy``, ``foggy``.
        source: ``"live"`` for an Open-Meteo answer, ``"offline"`` for the
            fictional demo dataset. The response generator must tell the user
            when this is ``"offline"``: a fictional forecast may never be
            presented as a real one.
        model_elevation: The elevation Open-Meteo actually used, when known.
            Useful for the report: it shows how far the grid cell sits from the
            summit the user asked about.
    """

    place_name: str
    state: str
    elevation: int | None
    target_date: date
    part_of_day: str | None
    temperature_max: float
    temperature_min: float
    weather_code: int
    cloud_cover: float
    wind_speed_max: float
    wind_gusts_max: float
    rain_sum: float
    snowfall_sum: float
    precipitation_probability_max: int
    sky: str
    source: str
    model_elevation: float | None = None

    @property
    def is_offline(self) -> bool:
        """Report whether these numbers are fictional demo data."""
        return self.source == SOURCE_OFFLINE


class ForecastOutOfRangeError(ValueError):
    """Raised when a day lies outside the forecast horizon.

    Separate from the network errors on purpose: this one is the user's
    question being unanswerable, not the service being unavailable, and the
    dialogue manager should say "I can only forecast 7 days ahead" rather than
    falling back to demo data.
    """


def describe_sky(weather_code: int, cloud_cover: float) -> str:
    """Map a WMO weather code and a cloud-cover percentage to a sky category.

    The rules, in order:

    1. **Fog first.** WMO 45 and 48 are fog; a foggy summit has no view
       regardless of how the cloud cover reads.
    2. **Precipitation implies cloud.** Any code from 51 (drizzle) upwards is
       reported as ``cloudy``, so the bot never says "sunny" about an hour
       that is also raining.
    3. **Otherwise the cloud cover decides**, on the okta convention:
       under 25 % is ``sunny``, 25-75 % is ``partly cloudy``, 75 % and above
       is ``cloudy``.

    Args:
        weather_code: WMO code.
        cloud_cover: Mean cloud cover in percent.

    Returns:
        One of the ``SKY_*`` categories.

    Examples:
        >>> describe_sky(0, 5.0)
        'sunny'
        >>> describe_sky(3, 90.0)
        'cloudy'
        >>> describe_sky(45, 10.0)
        'foggy'
        >>> describe_sky(61, 10.0)
        'cloudy'
    """
    if weather_code in FOG_CODES:
        return SKY_FOGGY
    if weather_code >= FIRST_PRECIPITATION_CODE:
        return SKY_CLOUDY
    if cloud_cover < CLOUD_SUNNY_BELOW:
        return SKY_SUNNY
    if cloud_cover >= CLOUD_CLOUDY_FROM:
        return SKY_CLOUDY
    return SKY_PARTLY_CLOUDY


# --- In-memory cache -------------------------------------------------------
# A plain module-level dict rather than functools.lru_cache, because entries
# have to expire: a cached forecast is useful for the next few minutes and
# wrong by tomorrow. Keyed by rounded coordinates plus elevation, so two
# lookups of the same place share one request.
_cache: dict[tuple[float, float, int], tuple[float, dict[str, Any]]] = {}


def clear_cache() -> None:
    """Empty the forecast cache.

    Used by the tests, so one test's cached payload cannot satisfy the next
    test's request and hide a bug.
    """
    _cache.clear()


def _cache_key(place: Place) -> tuple[float, float, int]:
    """Build the cache key for a place.

    Args:
        place: The place.

    Returns:
        Rounded latitude, longitude and elevation.
    """
    return (round(place.lat, 3), round(place.lon, 3), int(place.elevation or 0))


def fetch_payload(place: Place, need_hourly: bool) -> dict[str, Any]:
    """Request the raw forecast payload from Open-Meteo.

    Retries once on any network-level failure, then gives up and lets the
    caller fall back to the offline dataset.

    Args:
        place: The place to fetch, including its elevation.
        need_hourly: Whether hourly variables are needed as well. Requested
            only when a part of day has to be computed, to keep the response
            small.

    Returns:
        The decoded JSON payload.

    Raises:
        RuntimeError: If every attempt failed.
    """
    import requests  # imported lazily so the module imports without network deps

    key = _cache_key(place)
    cached = _cache.get(key)
    if cached is not None:
        stored_at, payload = cached
        # A payload fetched without hourly data cannot serve a part-of-day
        # question, so it is only reused when it carries what is needed.
        if time.monotonic() - stored_at < CACHE_TTL_SECONDS and (
            not need_hourly or "hourly" in payload
        ):
            return payload

    params: dict[str, Any] = {
        "latitude": place.lat,
        "longitude": place.lon,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "Europe/Vienna",
        # One extra day, because the "night" window runs past midnight into
        # the following day.
        "forecast_days": MAX_DAY_OFFSET + 2,
        **REQUEST_UNITS,
    }
    if place.elevation is not None:
        params["elevation"] = place.elevation
    if need_hourly:
        params["hourly"] = ",".join(HOURLY_VARIABLES)

    last_error: Exception | None = None
    for _ in range(MAX_ATTEMPTS):
        try:
            response = requests.get(
                API_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            payload = response.json()
            if "daily" not in payload:
                raise ValueError("response carried no 'daily' block")
            _cache[key] = (time.monotonic(), payload)
            return payload
        except Exception as error:  # noqa: BLE001 - every failure means fallback
            last_error = error

    raise RuntimeError(f"Open-Meteo request failed: {last_error}")


def _daily_index(payload: dict[str, Any], target: date) -> int:
    """Find the index of a date in the payload's daily block.

    Args:
        payload: Decoded API payload.
        target: The date wanted.

    Returns:
        The index into every ``daily`` list.

    Raises:
        ForecastOutOfRangeError: If the date is not in the response.
    """
    times = payload["daily"]["time"]
    wanted = target.isoformat()
    if wanted not in times:
        raise ForecastOutOfRangeError(
            f"{wanted} is outside the forecast range {times[0]}..{times[-1]}"
        )
    return times.index(wanted)


def _hourly_window(
    payload: dict[str, Any], target: date, part_of_day: str
) -> list[int]:
    """Collect the hourly indices belonging to one part of one day.

    Args:
        payload: Decoded API payload, including an ``hourly`` block.
        target: The day asked about.
        part_of_day: One of the ``PART_*`` constants.

    Returns:
        Indices into the hourly lists. Possibly fewer than the nominal window
        length at the end of the forecast range, where the data runs out.
    """
    start_hour, end_hour = PART_OF_DAY_HOURS[part_of_day]
    # Hours at or beyond 24 belong to the following day; this is what lets the
    # night window cross midnight.
    window_start = datetime.combine(target, datetime.min.time()) + timedelta(
        hours=start_hour
    )
    window_end = datetime.combine(target, datetime.min.time()) + timedelta(
        hours=end_hour
    )

    indices: list[int] = []
    for index, stamp in enumerate(payload["hourly"]["time"]):
        moment = datetime.fromisoformat(stamp)
        if window_start <= moment < window_end:
            indices.append(index)
    return indices


def _aggregate_hourly(
    payload: dict[str, Any], indices: list[int]
) -> dict[str, float]:
    """Reduce a set of hourly values to the same figures the daily block gives.

    How each variable is reduced, and why:

    * temperature - max and min over the window, as the daily block does.
    * wind and gusts - the **maximum**, because what matters is the strongest
      gust someone will be exposed to, not the average.
    * rain and snow - **summed**, since they are per-hour amounts.
    * probability - the maximum, matching ``precipitation_probability_max``.
    * cloud cover - the **mean**, matching ``cloud_cover_mean``.
    * weather code - the **highest** code in the window. WMO codes are
      ordered roughly by severity, so this answers "the worst it gets in that
      window", which is what someone planning a tour needs. The ordering is
      only approximate, which is why the code is used for the sky category and
      never printed as a number.

    Args:
        payload: Decoded API payload with an ``hourly`` block.
        indices: Hourly indices to aggregate.

    Returns:
        A dict with the same keys the daily path produces.
    """
    hourly = payload["hourly"]

    def values(name: str) -> list[float]:
        """Pull the non-null hourly values at the selected indices."""
        series = hourly[name]
        return [series[i] for i in indices if series[i] is not None]

    temperatures = values("temperature_2m") or [0.0]
    cloud = values("cloud_cover") or [0.0]
    codes = values("weather_code") or [0]

    return {
        "temperature_2m_max": max(temperatures),
        "temperature_2m_min": min(temperatures),
        "weather_code": int(max(codes)),
        "cloud_cover_mean": sum(cloud) / len(cloud),
        "wind_speed_10m_max": max(values("wind_speed_10m") or [0.0]),
        "wind_gusts_10m_max": max(values("wind_gusts_10m") or [0.0]),
        "rain_sum": sum(values("rain")),
        "snowfall_sum": sum(values("snowfall")),
        "precipitation_probability_max": int(
            max(values("precipitation_probability") or [0])
        ),
    }


def _report_from_values(
    place: Place,
    day: DaySlot,
    values: dict[str, float],
    source: str,
    model_elevation: float | None = None,
) -> WeatherReport:
    """Assemble a ``WeatherReport`` from a dict of raw figures.

    One construction point for both the live and the offline path, so the two
    can never drift into reporting different fields or a different sky rule.

    Args:
        place: The place.
        day: The resolved day.
        values: Raw figures, keyed as in the API's daily block.
        source: ``SOURCE_LIVE`` or ``SOURCE_OFFLINE``.
        model_elevation: Elevation the API used, when known.

    Returns:
        The finished report.
    """
    weather_code = int(values["weather_code"])
    cloud_cover = float(values["cloud_cover_mean"])

    return WeatherReport(
        place_name=place.name,
        state=place.state,
        elevation=place.elevation,
        target_date=day.target_date,
        part_of_day=day.part_of_day,
        temperature_max=round(float(values["temperature_2m_max"]), 1),
        temperature_min=round(float(values["temperature_2m_min"]), 1),
        weather_code=weather_code,
        cloud_cover=round(cloud_cover),
        wind_speed_max=round(float(values["wind_speed_10m_max"]), 1),
        wind_gusts_max=round(float(values["wind_gusts_10m_max"]), 1),
        rain_sum=round(float(values["rain_sum"]), 1),
        snowfall_sum=round(float(values["snowfall_sum"]), 1),
        precipitation_probability_max=int(values["precipitation_probability_max"]),
        sky=describe_sky(weather_code, cloud_cover),
        source=source,
        model_elevation=model_elevation,
    )


# --- Offline fallback ------------------------------------------------------


def load_offline_dataset(json_path: Path = OFFLINE_JSON) -> dict[str, Any]:
    """Read the fictional demo dataset.

    Args:
        json_path: Path to ``offline_forecasts.json``.

    Returns:
        The decoded dataset.

    Raises:
        FileNotFoundError: If the file is missing.
    """
    if not json_path.exists():
        raise FileNotFoundError(
            f"No offline dataset at {json_path}. It is part of the repository; "
            "if it is missing, the checkout is incomplete."
        )
    return json.loads(json_path.read_text(encoding="utf-8"))


def _offline_profile(dataset: dict[str, Any], place: Place) -> list[dict[str, Any]]:
    """Choose the offline day series to use for a place.

    A named entry is preferred, so the eight demo places have their own
    coherent little forecast. Any other place falls back to an
    **elevation-banded** generic profile, because elevation is what decides
    whether a plausible answer is +12 °C and rain or -6 °C and snow. Without
    that, the offline mode would answer a question about a 3 000 m summit with
    valley numbers, which is exactly the kind of implausibility that makes a
    demo fail in front of an audience.

    Args:
        dataset: The decoded dataset.
        place: The place asked about.

    Returns:
        A list of seven day dicts, offsets 0 to 6.
    """
    named = dataset["places"].get(place.name)
    if named is not None:
        return named["days"]

    elevation = place.elevation or 0
    for band in dataset["elevation_profiles"]:
        if band["min_elevation"] <= elevation <= band["max_elevation"]:
            return band["days"]

    # The bands cover 0..9000 m, so this is unreachable in practice; the
    # highest band is returned rather than raising, because the offline path
    # exists precisely to always produce an answer.
    return dataset["elevation_profiles"][-1]["days"]


def offline_forecast(
    place: Place, day: DaySlot, json_path: Path = OFFLINE_JSON
) -> WeatherReport:
    """Build a report from the fictional demo dataset.

    The dataset is stored **by day offset, not by date**, so the demo never
    goes stale: offset 0 is always "today", whenever the project is run.

    When a part of day was asked for, the daily figures are adjusted by a
    documented offset rather than invented per hour - keeping 7 days x 4 parts
    of hand-written values coherent would be busywork, and the offline path
    only has to be *plausible*, not accurate.

    Args:
        place: The place.
        day: The resolved day.
        json_path: Dataset location; overridden in tests.

    Returns:
        A report with ``source="offline"``.

    Raises:
        ForecastOutOfRangeError: If the day is outside the horizon.
    """
    if not 0 <= day.offset_days <= MAX_DAY_OFFSET:
        raise ForecastOutOfRangeError(
            f"offset {day.offset_days} is outside the offline range "
            f"0..{MAX_DAY_OFFSET}"
        )

    dataset = load_offline_dataset(json_path)
    values = dict(_offline_profile(dataset, place)[day.offset_days])

    if day.part_of_day is not None:
        adjustment = dataset["part_of_day_adjustments"][day.part_of_day]
        # Mornings and nights are colder than the daily maximum, afternoons
        # are the warmest part; wind usually peaks in the afternoon. The
        # factors live in the dataset so they can be read without the code.
        values["temperature_2m_max"] += adjustment["temperature_max_delta"]
        values["temperature_2m_min"] += adjustment["temperature_min_delta"]
        values["wind_speed_10m_max"] *= adjustment["wind_factor"]
        values["wind_gusts_10m_max"] *= adjustment["wind_factor"]
        # A part of a day gets a share of the day's precipitation.
        values["rain_sum"] *= adjustment["precipitation_factor"]
        values["snowfall_sum"] *= adjustment["precipitation_factor"]

    return _report_from_values(place, day, values, SOURCE_OFFLINE)


# --- The public entry point ------------------------------------------------


def get_forecast(
    place: Place,
    day: DaySlot,
    json_path: Path = OFFLINE_JSON,
    allow_live: bool = True,
) -> WeatherReport:
    """Get the forecast for a place and a day, live if possible.

    This is the function the dialogue manager calls. It never raises for a
    network problem: any failure produces the offline report instead, flagged
    with ``source="offline"``.

    Args:
        place: The place, carrying coordinates and elevation.
        day: The resolved day, possibly with a part of day.
        json_path: Offline dataset location; overridden in tests.
        allow_live: Set to ``False`` to force the offline path, which is what
            the demo mode and most tests use.

    Returns:
        The report, with ``source`` saying where the numbers came from.

    Raises:
        ForecastOutOfRangeError: If the day is beyond the forecast horizon.
            Deliberately *not* caught: the honest answer is "I can only
            forecast 7 days ahead", and silently serving demo data instead
            would be worse than saying so.
    """
    if not day.in_range:
        raise ForecastOutOfRangeError(
            f"{day.target_date} is {day.offset_days} days out; "
            f"the forecast covers {MAX_DAY_OFFSET + 1} days"
        )

    if allow_live:
        try:
            need_hourly = day.part_of_day is not None
            payload = fetch_payload(place, need_hourly=need_hourly)
            index = _daily_index(payload, day.target_date)

            if need_hourly and "hourly" in payload:
                indices = _hourly_window(payload, day.target_date, day.part_of_day)
                if indices:
                    values = _aggregate_hourly(payload, indices)
                else:
                    # The window fell outside the data (last day of the range
                    # for a night question). The whole day is a worse answer
                    # than no answer would be, but it is still the right one:
                    # better a slightly wider window than demo data.
                    values = {
                        name: payload["daily"][name][index]
                        for name in DAILY_VARIABLES
                    }
            else:
                values = {
                    name: payload["daily"][name][index] for name in DAILY_VARIABLES
                }

            return _report_from_values(
                place,
                day,
                values,
                SOURCE_LIVE,
                model_elevation=payload.get("elevation"),
            )
        except ForecastOutOfRangeError:
            raise
        except Exception:  # noqa: BLE001 - any live failure means fall back
            pass

    return offline_forecast(place, day, json_path=json_path)
