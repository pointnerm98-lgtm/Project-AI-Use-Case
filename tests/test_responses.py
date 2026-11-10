"""Tests for the response generator.

What is worth pinning here, and why:

* **The numbers in the sentence are the numbers from the report.** That is the
  reason the answers are templated rather than generated, so it deserves a
  test rather than trust.
* **Snow before rain when the question mentioned snow.** This is the step 2
  decision made good: rain and snow deliberately share one intent, and the
  ordering was deferred to this module.
* **The display name, never the canonical one.** Answering "Kepa" for the
  Mittagskogel would read as a different mountain.
* **The hint thresholds**, checked on their boundaries, because a hint that
  fires on everything is worse than no hint.
"""

from __future__ import annotations

import random
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from peakcast.intent_classifier import INTENT_LABELS
from peakcast.responses import (
    FRESH_SNOW_SUBSTANTIAL_CM,
    GUST_NOTABLE_KMH,
    GUST_STRONG_KMH,
    HARD_FREEZE_C,
    OFFLINE_NOTE,
    OFFLINE_NOTE_BRIEF,
    TEMPLATES,
    THUNDERSTORM_CODES,
    VALUES_STATED_BY_INTENT,
    build_hints,
    describe_day,
    describe_place,
    describe_precipitation,
    format_calendar_date,
    generate_answer,
    mentions_snow,
    preposition_for,
)
from peakcast.slots import PART_AFTERNOON, DaySlot, Place
from peakcast.weather import (
    SKY_CLOUDY,
    SKY_FOGGY,
    SKY_SUNNY,
    SOURCE_LIVE,
    SOURCE_OFFLINE,
    WeatherReport,
)

TODAY = date(2026, 11, 5)   # a Thursday

INNSBRUCK = Place(
    name="Innsbruck", state="Tirol", lat=47.27, lon=11.40, elevation=570,
    population=132493, feature_code="PPLA", display_name="Innsbruck",
)

GLOCKNER = Place(
    name="Großglockner", state="Tirol", lat=47.07, lon=12.69, elevation=3798,
    population=0, feature_code="MT", display_name="Großglockner",
)

# The case the display-name mechanism exists for: GeoNames files this mountain
# under its Slovene name.
MITTAGSKOGEL = Place(
    name="Kepa", state="Kärnten", lat=46.52, lon=14.07, elevation=2143,
    population=0, feature_code="MT", display_name="Mittagskogel",
    matched_text="Mittagskogel", ambiguous=True,
)


def a_day(offset: int = 0, part: str | None = None) -> DaySlot:
    """Build a day slot for the tests.

    Args:
        offset: Days from ``TODAY``.
        part: Part-of-day flag.

    Returns:
        The day slot.
    """
    return DaySlot(
        target_date=TODAY + timedelta(days=offset),
        offset_days=offset,
        part_of_day=part,
    )


def a_report(**overrides) -> WeatherReport:
    """Build a report with plausible defaults, overriding named fields.

    Args:
        **overrides: Fields to replace.

    Returns:
        The report.
    """
    values = {
        "place_name": "Innsbruck",
        "state": "Tirol",
        "elevation": 570,
        "target_date": TODAY,
        "part_of_day": None,
        "temperature_max": 9.4,
        "temperature_min": 3.3,
        "weather_code": 3,
        "cloud_cover": 80.0,
        "wind_speed_max": 12.0,
        "wind_gusts_max": 20.0,
        "rain_sum": 0.0,
        "snowfall_sum": 0.0,
        "precipitation_probability_max": 10,
        "sky": SKY_CLOUDY,
        "source": SOURCE_LIVE,
    }
    values.update(overrides)
    return WeatherReport(**values)


SEEDED = random.Random(42)


# --- The contract with the classifier ------------------------------------


def test_every_weather_intent_has_templates() -> None:
    """A weather intent with no template would fall back silently.

    The fallback is the overview template, which would answer the wrong
    question rather than crash - so this is checked explicitly.
    """
    weather_intents = {
        label for label in INTENT_LABELS if label.startswith("ask_")
    }
    assert weather_intents == set(TEMPLATES)


def test_templates_use_only_known_placeholders() -> None:
    """A typo'd placeholder would raise at answer time, in front of a user."""
    import string

    allowed = {
        "place", "in_place", "in_place_cap", "day", "day_cap",
        "tmin", "tmax", "sky", "cloud", "wind", "gusts", "precipitation",
    }
    for intent, variants in TEMPLATES.items():
        for template in variants:
            used = {
                name
                for _, name, _, _ in string.Formatter().parse(template)
                if name
            }
            assert used <= allowed, f"{intent}: unknown placeholder in {used}"


# --- Values land in the sentence -----------------------------------------


def test_the_numbers_in_the_answer_come_from_the_report() -> None:
    """Templating exists so the sentence cannot drift from the data."""
    report = a_report(
        temperature_min=-4.0, temperature_max=1.0, wind_gusts_max=55.0
    )
    answer = generate_answer(
        "ask_temperature", report, GLOCKNER, a_day(), TODAY, rng=SEEDED
    )

    assert "-4 °C" in answer
    assert "1 °C" in answer


def test_wind_answer_reports_both_mean_and_gusts() -> None:
    """Gusts are what decide whether a paraglider launches, per the concept."""
    report = a_report(wind_speed_max=18.0, wind_gusts_max=47.0)
    answer = generate_answer(
        "ask_wind", report, GLOCKNER, a_day(1), TODAY, rng=SEEDED
    )

    assert "18 km/h" in answer
    assert "47 km/h" in answer


# --- The display name ----------------------------------------------------


def test_the_answer_uses_the_display_name_not_the_canonical_one() -> None:
    """The Mittagskogel must not be answered as "Kepa".

    GeoNames files it under its Slovene name. Echoing that back would read as
    a different mountain to an Austrian user, which is the whole reason Place
    carries a display name.
    """
    answer = generate_answer(
        "ask_sky", a_report(sky=SKY_SUNNY), MITTAGSKOGEL, a_day(), TODAY,
        rng=SEEDED,
    )

    assert "Mittagskogel" in answer
    assert "Kepa" not in answer


def test_the_state_is_named_only_when_the_name_is_ambiguous() -> None:
    """Several real places answer to some names; one state per answer is enough.

    Adding the state to every answer would be noise, so it appears only when
    the name the user typed is genuinely shared.
    """
    assert describe_place(MITTAGSKOGEL) == "Mittagskogel (Kärnten)"
    assert describe_place(INNSBRUCK) == "Innsbruck"


@pytest.mark.parametrize(
    "feature_code, expected",
    [
        ("PPLA", "in"),       # you are *in* a town
        ("PPL", "in"),
        ("MT", "on the"),     # but *on* a mountain
        ("PK", "on the"),
        ("GLCR", "on the"),
        ("PASS", "at the"),
        ("HUT", "at the"),
        ("LK", "at the"),
        ("VAL", "in the"),
    ],
)
def test_preposition_follows_the_feature_type(
    feature_code: str, expected: str
) -> None:
    """"in Großglockner" would sound like the bot has never been outside.

    English picks the preposition from what kind of place it is, and the
    gazetteer already records that in the feature code.
    """
    place = Place(
        name="X", state="Tirol", lat=47.0, lon=11.0, elevation=1000,
        population=0, feature_code=feature_code,
    )
    assert preposition_for(place) == expected


# --- The day in words ----------------------------------------------------


@pytest.mark.parametrize(
    "offset, expected",
    [
        (0, "today"),
        (1, "tomorrow"),
        (2, "the day after tomorrow"),
        (3, "on Sunday"),
    ],
)
def test_day_is_described_the_way_a_person_would_say_it(
    offset: int, expected: str
) -> None:
    """Echoing "2026-11-08" back at the user would read like a database."""
    assert describe_day(a_day(offset), TODAY) == expected


def test_part_of_day_is_appended() -> None:
    """A part-of-day question must be answered about that part."""
    assert describe_day(a_day(1, PART_AFTERNOON), TODAY) == (
        "tomorrow in the afternoon"
    )


def test_weekday_can_be_rendered_without_its_preposition() -> None:
    """Sentences that supply their own preposition must not get a second one.

    This is what stops the out-of-range reply reading "about on Friday".
    """
    assert describe_day(a_day(3), TODAY, with_preposition=False) == "Sunday"


def test_calendar_date_is_spelled_out_unambiguously() -> None:
    """Used where "Friday" alone is not enough.

    A question beyond the forecast horizon is exactly that case: "next Friday"
    is ambiguous in English, so the bot has to show which Friday it meant.
    The day number is not zero-padded, because that is how a person writes it.
    """
    assert format_calendar_date(date(2026, 11, 13)) == "Friday 13 November"
    assert format_calendar_date(date(2026, 11, 3)) == "Tuesday 3 November"


# --- Precipitation ordering: the step 2 decision -------------------------


@pytest.mark.parametrize(
    "message, expected",
    [
        ("will it snow in Soelden tomorrow", True),
        ("how much fresh snow on the Dachstein", True),
        ("is there powder up there", True),
        ("any good for a ski tour tomorrow", True),
        ("will it rain in Graz tomorrow", False),
        ("do I need a rain jacket", False),
        ("hows the weather", False),
    ],
)
def test_snow_words_are_detected(message: str, expected: bool) -> None:
    """The snow/rain ordering is driven by the user's own wording."""
    assert mentions_snow(message) is expected


def test_snow_is_reported_first_when_the_question_was_about_snow() -> None:
    """The step 2 decision, made good here.

    Rain and snow deliberately share one intent, because the same sentence
    means rain in July and snow in January - the distinction is seasonal, not
    linguistic. So the ordering was deferred to the response generator, and
    this is it.
    """
    report = a_report(rain_sum=4.0, snowfall_sum=12.0)

    snow_question = describe_precipitation(report, snow_first=True)
    rain_question = describe_precipitation(report, snow_first=False)

    assert snow_question.index("snow") < snow_question.index("rain")
    assert rain_question.index("rain") < rain_question.index("snow")


def test_snow_ordering_runs_end_to_end_from_the_message() -> None:
    """generate_answer must take the ordering from the message it is given."""
    report = a_report(rain_sum=4.0, snowfall_sum=12.0)
    answer = generate_answer(
        "ask_precipitation", report, GLOCKNER, a_day(1), TODAY,
        message="how much fresh snow on the Glockner tomorrow", rng=SEEDED,
    )

    assert answer.index("snow") < answer.index("rain")


def test_a_dry_day_is_called_dry_not_zero_millimetres() -> None:
    """"0 mm of rain" reads as a measurement; "dry" is the answer."""
    phrase = describe_precipitation(
        a_report(rain_sum=0.0, snowfall_sum=0.0, precipitation_probability_max=5),
        snow_first=False,
    )
    assert phrase == "dry"
    assert "0 mm" not in phrase


def test_a_dry_day_with_a_real_chance_of_showers_says_so() -> None:
    """"Dry" and "dry but 60 % chance of showers" are different plans."""
    phrase = describe_precipitation(
        a_report(rain_sum=0.0, snowfall_sum=0.0, precipitation_probability_max=60),
        snow_first=False,
    )
    assert "60 %" in phrase


# --- Hints ---------------------------------------------------------------


def test_strong_gusts_produce_a_flying_hint() -> None:
    """Gusts are the concept's named decision variable for paragliders."""
    hints = build_hints(a_report(wind_gusts_max=GUST_STRONG_KMH), snow_first=False)
    assert hints
    assert "not flying weather" in hints[0]


def test_moderate_gusts_produce_a_milder_hint() -> None:
    """Just below the strong threshold the hint must soften, not vanish."""
    hints = build_hints(
        a_report(wind_gusts_max=GUST_STRONG_KMH - 1.0), snow_first=False
    )
    assert hints
    assert "worth checking" in hints[0]


def test_calm_weather_produces_no_gust_hint() -> None:
    """A hint that fires on every answer teaches the user to ignore it."""
    hints = build_hints(
        a_report(wind_gusts_max=GUST_NOTABLE_KMH - 1.0), snow_first=False
    )
    assert not any("gust" in hint.lower() for hint in hints)


def test_thunderstorms_outrank_every_other_hint() -> None:
    """The one hint closest to a warning must not be crowded out.

    Being caught on a ridge in a thunderstorm is the classic alpine accident,
    so it wins even against strong gusts and deep cold.
    """
    report = a_report(
        weather_code=max(THUNDERSTORM_CODES),
        wind_gusts_max=90.0,
        temperature_min=-20.0,
    )
    hints = build_hints(report, snow_first=False)

    assert "Thunderstorm" in hints[0]


def test_substantial_fresh_snow_points_at_the_avalanche_bulletin() -> None:
    """Deep new snow is exactly when the bot must defer to the real service."""
    hints = build_hints(
        a_report(snowfall_sum=FRESH_SNOW_SUBSTANTIAL_CM), snow_first=True
    )
    assert any("avalanche bulletin" in hint for hint in hints)


def test_fog_hint_is_about_the_view() -> None:
    """For this use case fog is a visibility answer, not a cloud one."""
    hints = build_hints(a_report(sky=SKY_FOGGY), snow_first=False)
    assert any("view" in hint for hint in hints)


def test_hard_freeze_produces_a_layers_hint() -> None:
    """Well below freezing is a gear question."""
    hints = build_hints(a_report(temperature_min=HARD_FREEZE_C), snow_first=False)
    assert any("below freezing" in hint for hint in hints)


def test_only_one_hint_is_appended_to_an_answer() -> None:
    """Two hints is a lecture. The list is ordered so the worst one survives."""
    report = a_report(
        weather_code=95, wind_gusts_max=90.0, temperature_min=-20.0,
        sky=SKY_FOGGY, snowfall_sum=30.0,
    )
    answer = generate_answer(
        "ask_overview", report, GLOCKNER, a_day(), TODAY, rng=SEEDED
    )

    assert len(build_hints(report, snow_first=False)) > 1
    assert answer.count("Thunderstorm") == 1
    assert "not flying weather" not in answer


# --- Hints must not repeat what the sentence already said ----------------


def test_gust_hint_drops_the_figure_when_the_answer_already_gave_it() -> None:
    """A wind answer already states the gusts, so the hint must not repeat them.

    Before this, an answer read: "...with gusts up to 29 km/h. Gusts reach
    29 km/h, worth checking before you launch." The advice is kept; only the
    duplicated number goes.
    """
    report = a_report(wind_speed_max=14.0, wind_gusts_max=29.0)
    answer = generate_answer(
        "ask_wind", report, GLOCKNER, a_day(1), TODAY, rng=SEEDED
    )

    assert "29 km/h" in answer, "the sentence itself must still give the value"
    assert answer.count("29 km/h") == 1, "but only once"
    assert "Worth checking before you launch." in answer


def test_gust_hint_keeps_the_figure_when_the_answer_does_not_give_it() -> None:
    """A temperature answer says nothing about wind, so the figure is news."""
    report = a_report(wind_gusts_max=29.0)
    answer = generate_answer(
        "ask_temperature", report, GLOCKNER, a_day(1), TODAY, rng=SEEDED
    )

    assert "Gusts reach 29 km/h" in answer


def test_strong_gust_hint_also_drops_a_duplicated_figure() -> None:
    """The stronger variant must de-duplicate too, not only the mild one."""
    report = a_report(wind_speed_max=60.0, wind_gusts_max=102.0)
    answer = generate_answer(
        "ask_overview", report, GLOCKNER, a_day(1), TODAY, rng=SEEDED
    )

    assert answer.count("102 km/h") == 1
    assert "not flying weather" in answer


def test_snow_hint_drops_the_figure_when_the_answer_already_gave_it() -> None:
    """A precipitation answer states the snow depth, so the hint must not."""
    report = a_report(snowfall_sum=22.0, precipitation_probability_max=90)
    answer = generate_answer(
        "ask_precipitation", report, GLOCKNER, a_day(1), TODAY,
        message="how much snow tomorrow", rng=SEEDED,
    )

    assert answer.count("22 cm") == 1
    assert "avalanche bulletin" in answer


def test_every_weather_intent_has_a_stated_values_entry() -> None:
    """A missing entry would silently reintroduce the duplication.

    ``VALUES_STATED_BY_INTENT.get(intent, frozenset())`` defaults to "states
    nothing", which is the safe direction but would quietly undo this fix for
    a new intent, so the map is required to be complete.
    """
    assert set(VALUES_STATED_BY_INTENT) == set(TEMPLATES)


def test_hints_can_be_switched_off() -> None:
    """The flag exists so a caller can get just the facts."""
    report = a_report(wind_gusts_max=90.0)
    answer = generate_answer(
        "ask_wind", report, GLOCKNER, a_day(), TODAY, rng=SEEDED,
        include_hint=False,
    )
    assert "not flying weather" not in answer


# --- Offline disclosure --------------------------------------------------


def test_offline_report_is_disclosed_in_full() -> None:
    """A fictional forecast must never read as a real one."""
    answer = generate_answer(
        "ask_temperature", a_report(source=SOURCE_OFFLINE), INNSBRUCK,
        a_day(), TODAY, rng=SEEDED,
    )
    assert OFFLINE_NOTE in answer


def test_offline_disclosure_has_a_short_form() -> None:
    """Repeats stay disclosed without repeating a paragraph ten times."""
    answer = generate_answer(
        "ask_temperature", a_report(source=SOURCE_OFFLINE), INNSBRUCK,
        a_day(), TODAY, rng=SEEDED, brief_offline_note=True,
    )
    assert OFFLINE_NOTE_BRIEF in answer
    assert OFFLINE_NOTE not in answer


def test_a_live_report_carries_no_disclosure() -> None:
    """A real forecast must not be hedged as sample data."""
    answer = generate_answer(
        "ask_temperature", a_report(source=SOURCE_LIVE), INNSBRUCK,
        a_day(), TODAY, rng=SEEDED,
    )
    assert "sample" not in answer.lower()


# --- Determinism and variety --------------------------------------------


def test_the_same_seed_gives_the_same_wording() -> None:
    """The tests need determinism; the RNG is injected rather than patched."""
    report = a_report()
    first = generate_answer(
        "ask_overview", report, INNSBRUCK, a_day(), TODAY,
        rng=random.Random(7),
    )
    second = generate_answer(
        "ask_overview", report, INNSBRUCK, a_day(), TODAY,
        rng=random.Random(7),
    )
    assert first == second


def test_phrasing_actually_varies() -> None:
    """Two identical answers in a row make a bot feel broken."""
    report = a_report()
    rng = random.Random(3)
    seen = {
        generate_answer(
            "ask_overview", report, INNSBRUCK, a_day(), TODAY, rng=rng
        )
        for _ in range(30)
    }
    assert len(seen) > 1
