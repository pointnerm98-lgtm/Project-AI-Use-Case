"""Unit tests for place and day extraction.

Two groups, matching the two slots:

* **Place** - the tests concentrate on the cases where a naive longest-match
  gazetteer lookup fails: multi-word names, the ASCII spellings of umlauts,
  GeoNames' compound spellings, typos, and the place names that are also
  ordinary English words. The last group is the important one, because those
  are the failures that produce a *confident wrong answer* (a forecast for the
  village of Rain) rather than a visible miss.
* **Day** - every test passes an explicit reference date. Without it the suite
  would pass or fail depending on which weekday it is run on, which is exactly
  the kind of flakiness that makes a test suite worthless as evidence.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from peakcast.slots import (
    MAX_DAY_OFFSET,
    PART_AFTERNOON,
    PART_EVENING,
    PART_MORNING,
    PART_NIGHT,
    extract_slots,
    find_day,
    find_place,
    normalise_name,
)

# A fixed Wednesday. Every day test resolves against it, so "on Friday" is
# always two days out and "next Friday" always nine, run on any machine on any
# date.
WEDNESDAY = date(2026, 11, 4)


# --- Normalisation ---------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        # umlaut, its ASCII digraph and its bare vowel must all collapse
        ("Sölden", "solden"),
        ("Soelden", "solden"),
        ("Solden", "solden"),
        ("Kitzbühel", "kitzbuhel"),
        ("Kitzbuehel", "kitzbuhel"),
        ("Schmittenhöhe", "schmittenhohe"),
        ("Schmittenhoehe", "schmittenhohe"),
        # eszett
        ("Großglockner", "grossglockner"),
        ("Grossglockner", "grossglockner"),
        # "Sankt" and "St." are the same word
        ("St. Johann im Pongau", "st johann im pongau"),
        ("Sankt Johann im Pongau", "st johann im pongau"),
        # punctuation and case
        ("ZELL AM SEE", "zell am see"),
        ("Zell-am-See", "zell am see"),
    ],
)
def test_normalise_name_collapses_spelling_variants(raw: str, expected: str) -> None:
    """All spellings a user might type must reduce to one key."""
    assert normalise_name(raw) == expected


# --- Place: multi-word names ----------------------------------------------


@pytest.mark.parametrize(
    "message, expected_name",
    [
        # The headline case for longest-match: stopping at "Zell" would pick a
        # different place entirely.
        ("hows the weather in Zell am See tomorrow", "Zell am See"),
        ("weather in zell am see", "Zell am See"),
        ("ZELL AM SEE forecast", "Zell am See"),
        # "Sankt" vs "St." - GeoNames stores both forms.
        ("St. Johann im Pongau forecast", "Sankt Johann im Pongau"),
        ("Sankt Johann im Pongau", "Sankt Johann im Pongau"),
        # GeoNames stores this pass as a single word, the user writes two.
        ("how windy on the Turracher Hoehe", "Turracherhöhe"),
        ("snow at the Turracherhoehe", "Turracherhöhe"),
    ],
)
def test_multi_word_names_match_in_full(message: str, expected_name: str) -> None:
    """The longest matching name must win, not its first word."""
    place = find_place(message)
    assert place is not None, f"no place found in {message!r}"
    assert place.name == expected_name


# --- Place: umlauts and ASCII spellings ------------------------------------


@pytest.mark.parametrize(
    "message, expected_name",
    [
        ("weather in Soelden", "Sölden"),
        ("weather in Sölden", "Sölden"),
        ("Kitzbuehel snow report", "Kitzbühel"),
        ("Schmittenhoehe wind", "Schmittenhöhe"),
        ("conditions on the Hochkoenig", "Hochkönig"),
        ("Grossglockner wind tomorrow", "Großglockner"),
    ],
)
def test_ascii_spellings_of_umlauts_resolve(message: str, expected_name: str) -> None:
    """A user on a non-German keyboard must still be understood."""
    place = find_place(message)
    assert place is not None, f"no place found in {message!r}"
    assert place.name == expected_name


# --- Place: typos ----------------------------------------------------------


def test_small_typo_is_matched_fuzzily_and_flagged() -> None:
    """A near-miss must resolve, and must say that it was a near-miss."""
    place = find_place("temperature in Innsbruk")
    assert place is not None
    assert place.name == "Innsbruck"
    # The dialogue manager needs to know, so it can confirm back rather than
    # silently answering for a place the user did not type.
    assert place.fuzzy is True


def test_exact_match_is_not_flagged_as_fuzzy() -> None:
    """An exact hit must not claim to be a guess."""
    place = find_place("temperature in Innsbruck")
    assert place is not None
    assert place.name == "Innsbruck"
    assert place.fuzzy is False


@pytest.mark.parametrize(
    "message, expected_name",
    [
        # Users shorten compound names; GeoNames stores the long form.
        ("will I see the stars from the Krippenstein tonight", "Hoher Krippenstein"),
        ("whats the cloud base at the Bischling", "Bischlinghöhe"),
        ("gust activity at the Sonnwendjoch", "Hinteres Sonnwendjoch"),
    ],
)
def test_shortened_compound_names_resolve_to_the_full_name(
    message: str, expected_name: str
) -> None:
    """A correct but incomplete name must find the right place.

    This is not typo tolerance and must not be handled by it: fuzzy matching
    resolved "Bischling" to "Bichling" and "Sonnwendjoch" to "Sonnenjoch" -
    different real places, which is the worst kind of error because the answer
    looks entirely plausible.
    """
    place = find_place(message)
    assert place is not None, f"no place found in {message!r}"
    assert place.name == expected_name
    # Still an assumption, so it must be flagged for confirmation.
    assert place.fuzzy is True


def test_language_name_is_not_read_as_a_place() -> None:
    """A word English always capitalises must not pass the capital test.

    "can you speak German" matched the village of Germanns before words like
    this were added to the ambiguous list: "German" is a prefix of "Germanns"
    and is capitalised in every sentence it ever appears in, so the
    capitalisation evidence said nothing.
    """
    assert find_place("can you speak German") is None
    assert find_place("can you translate this to German") is None


def test_nonsense_word_is_not_forced_onto_a_place() -> None:
    """Fuzzy matching must refuse rather than invent.

    The conservative cut-off is the point of this test: a wrong place produces
    a plausible forecast for somewhere the user never asked about, which is
    worse than admitting the name was not recognised.
    """
    assert find_place("weather in Xyzzyquux") is None


# --- Place: names that are also English words ------------------------------
# The most dangerous group. Austria has 24 gazetteer entries named "Rain", a
# town called "Hard" with 13 495 inhabitants, and villages called "Going",
# "See", "Still", "First" and "Quick".


@pytest.mark.parametrize(
    "message",
    [
        "is it going to rain in Wien",
        "will it rain tomorrow",
        "is it going to rain",
        "is it hard to get up there",
        "I want to see the view",
        "is it still snowing",
        "whats the first thing to pack",
        "do I need a rain jacket",
    ],
)
def test_english_words_are_not_read_as_places(message: str) -> None:
    """An ordinary English word must not become a forecast location."""
    place = find_place(message)
    # Either nothing is found, or something else in the sentence is - but never
    # the English word itself.
    if place is not None:
        assert place.name not in {
            "Rain", "Hard", "Going", "See", "Still", "First", "Quick", "Pack", "Fall",
        }, f"{message!r} wrongly matched the place {place.name!r}"


def test_the_forst_regression() -> None:
    """The longest matching name must win over an earlier, shorter one.

    "Sankt" normalises to "st", so this sentence contains the window
    "for st", and removing its space gives "forst" - the village of Forst.
    A leftmost-first scan returned that and never reached the four-token
    "st anton am arlberg" further right.
    """
    place = find_place("forecast for Sankt Anton am Arlberg tonight")
    assert place is not None
    assert place.name == "St Anton am Arlberg"


def test_the_going_to_rain_regression() -> None:
    """"going to rain" must resolve to the real place in the sentence.

    This sentence is the reason "to" was removed from the place prepositions:
    with it, "to rain" looked like "to <place>" and the bot answered for the
    village of Rain in Kärnten.
    """
    place = find_place("is it going to rain in Wien")
    assert place is not None
    assert place.name == "Vienna"
    assert place.state == "Wien"


@pytest.mark.parametrize(
    "message, expected_name",
    [
        # With a preposition in front, the same words *are* places.
        ("whats the weather in Hard", "Hard"),
        ("weather in See", "See"),
        ("how cold in Going", "Going"),
        ("forecast for Rain", "Rain"),
    ],
)
def test_ambiguous_names_match_with_evidence(message: str, expected_name: str) -> None:
    """Ambiguous names must still work when the user clearly means the place.

    Dropping these names from the gazetteer was not an option: "Hard" is a real
    town of 13 495 people, and someone there will ask for its weather.
    """
    place = find_place(message)
    assert place is not None, f"no place found in {message!r}"
    assert place.name == expected_name


def test_capitalisation_alone_is_enough_evidence() -> None:
    """An initial capital is the second accepted kind of evidence."""
    place = find_place("Going am Wilden Kaiser snow")
    assert place is not None
    assert place.name == "Going"


# --- Place: ambiguity and absence -----------------------------------------


def test_ambiguous_name_resolves_to_the_most_populous() -> None:
    """Same-named places are decided by population."""
    place = find_place("weather in Zell am See")
    assert place is not None
    assert place.name == "Zell am See"
    # The town, not the much larger administrative district of the same name:
    # ADM rows are excluded by scripts/build_places.py precisely because their
    # elevation would be wrong for a forecast.
    assert place.population > 0
    assert place.elevation is not None and place.elevation < 1200


def test_place_carries_what_the_weather_service_needs() -> None:
    """A matched place must be usable as a forecast query."""
    place = find_place("weather on the Grossglockner")
    assert place is not None
    assert 46.0 < place.lat < 49.1          # Austria's latitude range
    assert 9.5 < place.lon < 17.2           # Austria's longitude range
    assert place.elevation is not None and place.elevation > 3000
    assert place.state == "Tirol"
    assert place.label == "Großglockner, Tirol"


def test_no_place_returns_none() -> None:
    """With no place named, the slot stays empty.

    Falling back to the place the user mentioned earlier is the dialogue
    manager's job, not this module's - see the module docstring.
    """
    assert find_place("hows the weather") is None
    assert find_place("temperature please") is None
    assert find_place("") is None


# --- Day: the closed vocabulary -------------------------------------------


@pytest.mark.parametrize(
    "message, expected_offset, expected_part",
    [
        ("hows the weather today", 0, None),
        ("right now", 0, None),
        ("whats it like tonight", 0, PART_NIGHT),
        ("this morning in Lienz", 0, PART_MORNING),
        ("this afternoon in Lienz", 0, PART_AFTERNOON),
        ("this evening", 0, PART_EVENING),
        ("tomorrow", 1, None),
        ("tomorrow morning", 1, PART_MORNING),
        ("tomorrow evening", 1, PART_EVENING),
        ("the day after tomorrow", 2, None),
        ("in 3 days", 3, None),
        # Wednesday reference -> Friday is two days out
        ("on Friday", 2, None),
        ("will it snow on sunday", 4, None),
        # the same weekday as the reference means today, not a week away
        ("whats up on Wednesday", 0, None),
        # the typo that is present in the intent training data
        ("tomorow", 1, None),
    ],
)
def test_day_words_resolve(
    message: str, expected_offset: int, expected_part: str | None
) -> None:
    """Each day word must resolve to the right offset and part of day."""
    day = find_day(message, reference=WEDNESDAY)
    assert day.offset_days == expected_offset, f"{message!r} -> {day.target_date}"
    assert day.part_of_day == expected_part
    assert day.in_range is True


def test_weekend_resolves_to_saturday() -> None:
    """A weekend spans two days, so one has to be chosen - Saturday."""
    day = find_day("hows the weather this weekend", reference=WEDNESDAY)
    assert day.target_date == date(2026, 11, 7)
    assert day.target_date.strftime("%A") == "Saturday"
    assert day.offset_days == 3


def test_default_is_today_and_says_so() -> None:
    """With no day word, today is assumed - and the flag records that."""
    day = find_day("how windy is it on the Hochkoenig", reference=WEDNESDAY)
    assert day.offset_days == 0
    assert day.target_date == WEDNESDAY
    assert day.is_default is True
    assert day.in_range is True


# --- Day: the forecast horizon --------------------------------------------


@pytest.mark.parametrize(
    "message, expected_offset",
    [
        ("next Friday", 9),      # the Friday of the following week
        ("in 10 days", 10),
        ("in a week", 7),
        ("in 2 weeks", 14),
    ],
)
def test_dates_beyond_the_horizon_are_flagged_not_clamped(
    message: str, expected_offset: int
) -> None:
    """Out-of-range days must be reported, not silently moved.

    Clamping would make the bot answer for the wrong day while sounding
    confident; the flag lets it say "I can only forecast 7 days ahead".
    """
    day = find_day(message, reference=WEDNESDAY)
    assert day.offset_days == expected_offset
    assert day.in_range is False
    assert day.offset_days > MAX_DAY_OFFSET


def test_the_last_day_in_range_is_still_in_range() -> None:
    """The boundary itself must be answerable."""
    day = find_day(f"in {MAX_DAY_OFFSET} days", reference=WEDNESDAY)
    assert day.offset_days == MAX_DAY_OFFSET
    assert day.in_range is True


# --- Both slots together ---------------------------------------------------


def test_extract_slots_returns_both() -> None:
    """The combined entry point must fill both slots from one message."""
    slots = extract_slots(
        "how windy is it on the Hochkoenig tomorrow", reference=WEDNESDAY
    )
    assert slots.place is not None
    assert slots.place.name == "Hochkönig"
    assert slots.day.offset_days == 1
    assert slots.day.is_default is False


def test_extract_slots_with_no_place_still_has_a_day() -> None:
    """A missing place must not stop the day from being resolved."""
    slots = extract_slots("whats it like tomorrow", reference=WEDNESDAY)
    assert slots.place is None
    assert slots.day.offset_days == 1
