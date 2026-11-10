"""Response generator: turn a WeatherReport into one or two sentences.

Position in the architecture: the last stage. The NLU has decided what was
asked, the slot extractor where and when, the weather service has the numbers.
This module is the only place that decides *wording*.

Design decisions
----------------

* **Templates, not a language model.** The concept rules out LLM APIs, and for
  this job templates are not a compromise: the answers have to contain exact
  values with units, and a template guarantees that the number in the sentence
  is the number the service returned. A generated sentence could not.

* **A few variants per intent, chosen with an injectable RNG.** Two identical
  answers in a row make a bot feel broken, so each intent has several
  phrasings. The RNG is a parameter rather than the module-level ``random``,
  so the tests are deterministic without having to monkeypatch anything.

* **Hints are rules with named thresholds.** The concept asks for a short
  outdoor hint, and the thresholds below are **heuristics chosen for this
  project, not official limits** - that is marked on every constant, because a
  number in a safety-adjacent sentence invites being read as authoritative.
  The bot informs; it does not certify conditions.

* **Offline data is always disclosed.** If the report came from the fictional
  demo dataset, the answer says so. A fictional forecast must never read as a
  real one, which is why ``WeatherReport`` carries its source at all.
"""

from __future__ import annotations

import random
from datetime import date

from peakcast.slots import (
    PART_AFTERNOON,
    PART_EVENING,
    PART_MORNING,
    PART_NIGHT,
    DaySlot,
    Place,
)
from peakcast.weather import (
    SKY_CLOUDY,
    SKY_FOGGY,
    SKY_PARTLY_CLOUDY,
    SKY_SUNNY,
    WeatherReport,
)

# --- Intent names ----------------------------------------------------------
# Repeated here rather than imported from the classifier module so this module
# can be read on its own; they are asserted against INTENT_LABELS in the tests.
ASK_TEMPERATURE: str = "ask_temperature"
ASK_SKY: str = "ask_sky"
ASK_WIND: str = "ask_wind"
ASK_PRECIPITATION: str = "ask_precipitation"
ASK_OVERVIEW: str = "ask_overview"

# --- Hint thresholds -------------------------------------------------------
# HEURISTICS for this student project, not official limits. They are set to be
# useful rather than cautious-at-any-cost: a hint that fires on every answer
# teaches the user to ignore it.
#
# Gusts: recreational paraglider pilots commonly treat the mid-20s km/h as the
# point where a launch needs thinking about and the mid-30s as "not today".
# Real decisions depend on the site, the wing and the pilot, which is exactly
# why the bot hints rather than advises.
GUST_NOTABLE_KMH: float = 25.0
GUST_STRONG_KMH: float = 40.0

# Fresh snow worth mentioning to a ski tourer, and enough to change the day.
FRESH_SNOW_NOTABLE_CM: float = 1.0
FRESH_SNOW_SUBSTANTIAL_CM: float = 15.0

# Rain worth taking a jacket for, in mm over the period.
RAIN_NOTABLE_MM: float = 1.0

# Below this, exposed skin and water bottles are the problem.
FREEZING_C: float = 0.0
HARD_FREEZE_C: float = -10.0

# A warm day on which someone will underestimate how much water to carry.
WARM_C: float = 25.0

# WMO codes for thunderstorms. The one weather type where the hint is close to
# a warning: being on a ridge in one is the classic alpine accident.
THUNDERSTORM_CODES: frozenset[int] = frozenset({95, 96, 99})

# Words that make a precipitation question a *snow* question. Checked against
# the user's own message, because the step 2 decision was to keep rain and snow
# in one intent and let this module order the answer: the same sentence means
# rain in July and snow in January, so the distinction is seasonal rather than
# linguistic and belongs here, not in the classifier.
SNOW_WORDS: frozenset[str] = frozenset(
    {
        "snow", "snowfall", "snowing", "snowy", "snwo", "snoww",
        "powder", "fresh", "freshies", "ski", "skiing", "skitour",
        "skitouring", "tour", "touring", "slopes", "piste", "pistes",
        "schnee", "neuschnee", "firn", "base", "cover",
    }
)

# How a part of day is referred to in a sentence.
PART_PHRASES: dict[str, str] = {
    PART_MORNING: "in the morning",
    PART_AFTERNOON: "in the afternoon",
    PART_EVENING: "in the evening",
    PART_NIGHT: "overnight",
}

# --- Which values the answer sentence already states ----------------------
# A hint that repeats a number the sentence just gave reads like a stutter:
#
#   "Wind on the Hochkoenig tomorrow is around 14 km/h, with gusts up to
#    29 km/h. Gusts reach 29 km/h, worth checking before you launch."
#
# So the hints are told which kinds of value the answer already carries, and
# drop the figure when it would be a repeat. They never drop the *advice* -
# only the number - because the advice is the part the user cannot derive.
#
# Keyed by intent, because the intent decides which template was used and
# therefore which values appear. `ask_overview` states almost everything,
# which is exactly where the stutter was most visible.
VALUE_GUSTS: str = "gusts"
VALUE_SNOW: str = "snow"

VALUES_STATED_BY_INTENT: dict[str, frozenset[str]] = {
    ASK_TEMPERATURE: frozenset(),
    ASK_SKY: frozenset(),
    ASK_WIND: frozenset({VALUE_GUSTS}),
    ASK_PRECIPITATION: frozenset({VALUE_SNOW}),
    ASK_OVERVIEW: frozenset({VALUE_GUSTS, VALUE_SNOW}),
}

# --- Prepositions ----------------------------------------------------------
# You are "in Innsbruck" but "on the Großglockner", and a bot that says "in
# Großglockner" sounds like it has never been outside. English picks the
# preposition from what kind of place it is, and the gazetteer already records
# exactly that in the GeoNames feature code, so the mapping is free.
#
# Keyed by feature code; the default covers populated places and anything
# unlisted, since "in" is the safe choice for a settlement.
PREPOSITION_BY_FEATURE: dict[str, str] = {
    # summits, ridges and rock: you are *on* them
    "MT": "on the",
    "PK": "on the",
    "MTS": "in the",      # a whole range: "in the Hohe Tauern"
    "HLL": "on the",
    "RDGE": "on the",
    "CLF": "on the",
    "CRQ": "in the",
    "SLP": "on the",
    "GLCR": "on the",
    # cols and saddles: "at the Turracherhöhe"
    "PASS": "at the",
    "SDL": "at the",
    # a valley or a named region encloses you
    "VAL": "in the",
    "RGN": "on the",      # "on the Raxalpe" - these are massifs in practice
    # places you walk to and stand at
    "HUT": "at the",
    "LK": "at the",
}
DEFAULT_PREPOSITION: str = "in"

# What the sky category looks like as a predicate.
SKY_PHRASES: dict[str, str] = {
    SKY_SUNNY: "sunny",
    SKY_PARTLY_CLOUDY: "partly cloudy",
    SKY_CLOUDY: "cloudy",
    SKY_FOGGY: "foggy",
}

# Added to any answer built from the demo dataset. A fictional forecast must
# never read as a real one, so this is not optional - but repeating the full
# sentence on all ten answers of a conversation turns it into wallpaper that a
# reader skips, which defeats the point. So the full note is used the first
# time and a short tag afterwards, which stays visible without crowding out
# the answer.
OFFLINE_NOTE: str = (
    "Heads-up: the live forecast service is unreachable right now, so these "
    "are sample values from the offline demo dataset, not a real forecast."
)
OFFLINE_NOTE_BRIEF: str = "(Sample data, not a real forecast.)"


def mentions_snow(message: str) -> bool:
    """Report whether the user's wording makes this a snow question.

    Args:
        message: The raw user message.

    Returns:
        ``True`` if any snow-related word appears.
    """
    words = {
        "".join(ch for ch in token if ch.isalpha())
        for token in message.lower().split()
    }
    return bool(words & SNOW_WORDS)


def describe_day(day: DaySlot, today: date, with_preposition: bool = True) -> str:
    """Put the day into words the way a person would say it.

    "today", "tomorrow" and a weekday name are what a user expects back;
    echoing "2026-11-07" would read like a database. The part of day is
    appended when there is one.

    Args:
        day: The resolved day.
        today: The date that counts as today, so the wording stays correct
            whenever the bot is run.
        with_preposition: Whether a weekday gets its "on". Sentences that
            already supply a preposition pass ``False``, which is what stops
            the out-of-range reply reading "I can't tell you about on Friday".

    Returns:
        A phrase such as ``"tomorrow"`` or ``"on Saturday in the afternoon"``.
    """
    offset = (day.target_date - today).days

    if offset == 0:
        words = "today"
    elif offset == 1:
        words = "tomorrow"
    elif offset == 2:
        words = "the day after tomorrow"
    else:
        # Within the forecast week a weekday name is unambiguous and friendlier
        # than a date.
        weekday = day.target_date.strftime("%A")
        words = f"on {weekday}" if with_preposition else weekday

    if day.part_of_day:
        words = f"{words} {PART_PHRASES[day.part_of_day]}"
    return words


def format_calendar_date(day: date) -> str:
    """Spell a date out as "Friday 13 November".

    Used where the bot has to be unambiguous rather than friendly. "Friday" on
    its own is not enough when the point is that the user and the bot may mean
    different Fridays - which is exactly the case for a question beyond the
    forecast horizon, where "next Friday" is genuinely ambiguous in English and
    the bot's reading of it should be visible.

    The day number is not zero-padded, because "Friday 3 November" is how a
    person writes it.

    Args:
        day: The date.

    Returns:
        The spelled-out date.
    """
    return f"{day.strftime('%A')} {day.day} {day.strftime('%B')}"


def describe_place(place: Place) -> str:
    """Name the place the way the user knows it.

    Uses the **display name**, never the canonical GeoNames name: the
    Mittagskogel is filed as *Kepa* and the Mölltaler Gletscher as
    *Wurtenkees*, and answering with those would read as a different mountain.

    The federal state is added only when the name was ambiguous - several real
    places answer to "See", "Annaberg" or "Mittagskogel", so the answer has to
    say which one it used. Adding the state to every answer would be noise.

    Args:
        place: The resolved place.

    Returns:
        ``"Sölden"`` or ``"Annaberg (Salzburg)"``.
    """
    if place.ambiguous and place.state:
        return f"{place.shown} ({place.state})"
    return place.shown


def preposition_for(place: Place) -> str:
    """Return the preposition that goes in front of this place.

    "in Innsbruck" but "on the Großglockner" - English chooses by what kind of
    place it is, and the GeoNames feature code already says that.

    Args:
        place: The resolved place.

    Returns:
        ``"in"``, ``"on the"`` or ``"at the"``.
    """
    return PREPOSITION_BY_FEATURE.get(place.feature_code, DEFAULT_PREPOSITION)


def format_temperature(value: float) -> str:
    """Format a temperature without a pointless trailing zero.

    Args:
        value: Temperature in °C.

    Returns:
        ``"-4 °C"`` or ``"8.5 °C"``.
    """
    if abs(value - round(value)) < 0.05:
        return f"{round(value):.0f} °C"
    return f"{value:.1f} °C"


def format_amount(value: float, unit: str) -> str:
    """Format a precipitation amount.

    Args:
        value: The amount.
        unit: ``"mm"`` or ``"cm"``.

    Returns:
        ``"9 mm"`` or ``"0.4 cm"``.
    """
    if abs(value - round(value)) < 0.05:
        return f"{round(value):.0f} {unit}"
    return f"{value:.1f} {unit}"


# --- Sentence templates ----------------------------------------------------
# Each intent has several phrasings. Placeholders are filled by
# ``_template_values``; every template uses only placeholders, so a number can
# never be hard-coded into a sentence by accident.
TEMPLATES: dict[str, tuple[str, ...]] = {
    ASK_TEMPERATURE: (
        "{place} {day}: {tmin} to {tmax}.",
        "Expect {tmin} to {tmax} {in_place} {day}.",
        "{day_cap} {in_place} it should be between {tmin} and {tmax}.",
    ),
    ASK_SKY: (
        "{place} {day}: {sky}, with {cloud} % cloud cover.",
        "It should be {sky} {in_place} {day} ({cloud} % cloud).",
        "{day_cap} {in_place} the sky looks {sky}, around {cloud} % cloud cover.",
    ),
    ASK_WIND: (
        "{place} {day}: wind around {wind}, gusting to {gusts}.",
        "Wind {in_place} {day} is around {wind}, with gusts up to {gusts}.",
        "{day_cap} {in_place} expect {wind} of wind and gusts to {gusts}.",
    ),
    ASK_PRECIPITATION: (
        "{place} {day}: {precipitation}.",
        "{day_cap} {in_place}: {precipitation}.",
        "{in_place_cap} {day} the forecast is {precipitation}.",
    ),
    ASK_OVERVIEW: (
        "{place} {day}: {sky}, {tmin} to {tmax}, wind to {gusts}, "
        "{precipitation}.",
        "{day_cap} {in_place} it looks {sky} with {tmin} to {tmax}, "
        "gusts to {gusts} and {precipitation}.",
        "{in_place_cap} {day}: {tmin} to {tmax}, {sky}, gusts to {gusts}, "
        "{precipitation}.",
    ),
}


def describe_precipitation(report: WeatherReport, snow_first: bool) -> str:
    """Put rain and snowfall into one phrase.

    Ordering is the step 2 decision made good: rain and snow share one intent,
    and the response generator reports **snowfall first when the question
    mentioned snow**, otherwise rainfall first. Amounts that round to zero are
    left out entirely rather than reported as "0 mm of rain", which reads as a
    measurement rather than as "dry".

    Args:
        report: The forecast.
        snow_first: Whether the user's wording made this a snow question.

    Returns:
        A phrase such as ``"12 cm of fresh snow (90 % chance)"`` or
        ``"dry"``.
    """
    parts: list[str] = []
    rain = (
        f"{format_amount(report.rain_sum, 'mm')} of rain"
        if report.rain_sum >= 0.05
        else ""
    )
    snow = (
        f"{format_amount(report.snowfall_sum, 'cm')} of fresh snow"
        if report.snowfall_sum >= 0.05
        else ""
    )

    if snow_first:
        parts = [p for p in (snow, rain) if p]
    else:
        parts = [p for p in (rain, snow) if p]

    if not parts:
        # No measurable precipitation. The probability still matters: "dry, but
        # a 40 % chance of showers" is a different plan from "dry".
        if report.precipitation_probability_max >= 30:
            return (
                "no measurable precipitation, though there is a "
                f"{report.precipitation_probability_max} % chance of showers"
            )
        return "dry"

    phrase = " and ".join(parts)
    if report.precipitation_probability_max:
        phrase += f" ({report.precipitation_probability_max} % chance)"
    return phrase


def build_hints(
    report: WeatherReport,
    snow_first: bool,
    stated: frozenset[str] = frozenset(),
) -> list[str]:
    """Produce the outdoor hints that apply to this forecast.

    Rules only, with the thresholds named at the top of the module and marked
    there as heuristics. Ordered by how much they should change someone's
    plan, so that when the list is trimmed the most consequential hint
    survives.

    Args:
        report: The forecast.
        snow_first: Whether the user asked about snow, which promotes the
            fresh-snow hint.
        stated: The kinds of value the answer sentence already gives (see
            ``VALUES_STATED_BY_INTENT``). A hint that would repeat one of them
            drops the figure and keeps the advice.

    Returns:
        Hint sentences, most important first.
    """
    hints: list[str] = []

    # 1. Thunderstorms. The closest this bot comes to a warning, because being
    #    caught on a ridge in one is the classic alpine accident.
    if report.weather_code in THUNDERSTORM_CODES:
        hints.append(
            "Thunderstorms are forecast - plan to be off summits and ridges early."
        )

    # 2. Gusts. The concept names gusts as the value that decides whether a
    #    paraglider launches, so this hint is phrased for that reader.
    gusts_already_said = VALUE_GUSTS in stated
    if report.wind_gusts_max >= GUST_STRONG_KMH:
        hints.append(
            "That is not flying weather, and exposed ridges will be unpleasant."
            if gusts_already_said
            else f"Gusts to {report.wind_gusts_max:.0f} km/h are strong - "
            "not flying weather, and exposed ridges will be unpleasant."
        )
    elif report.wind_gusts_max >= GUST_NOTABLE_KMH:
        hints.append(
            "Worth checking before you launch."
            if gusts_already_said
            else f"Gusts reach {report.wind_gusts_max:.0f} km/h, "
            "worth checking before you launch."
        )

    # 3. Fresh snow, promoted when the question was about snow.
    if report.snowfall_sum >= FRESH_SNOW_SUBSTANTIAL_CM:
        hints.append(
            "That much new snow changes the day - and it is also why the "
            "avalanche bulletin matters more than I do."
            if VALUE_SNOW in stated
            else f"{report.snowfall_sum:.0f} cm of new snow changes the day - "
            "and it is also why the avalanche bulletin matters more than I do."
        )
    elif report.snowfall_sum >= FRESH_SNOW_NOTABLE_CM and snow_first:
        hints.append("There is a little fresh snow on top of the old cover.")

    # 4. Fog. For this use case it is about the view and orientation, which is
    #    what the ask_sky intent is really asking about.
    if report.sky == SKY_FOGGY:
        hints.append("Fog is likely, so do not count on the view or on landmarks.")

    # 5. Cold. Below freezing is a gear question; well below is a real one.
    if report.temperature_min <= HARD_FREEZE_C:
        hints.append(
            "It is well below freezing - windproof layers and something warm "
            "to drink."
        )
    elif report.temperature_min <= FREEZING_C:
        hints.append(
            "Around or below freezing, so expect icy patches early and late."
        )

    # 6. Rain, only when there is no stronger hint already.
    if report.rain_sum >= RAIN_NOTABLE_MM and not hints:
        hints.append("Take a rain jacket.")

    # 7. Heat, the one hint that is not about cold or wind.
    if report.temperature_max >= WARM_C:
        hints.append("It will be warm - carry more water than you think you need.")

    return hints


def _template_values(
    report: WeatherReport, place: Place, day: DaySlot, today: date, snow_first: bool
) -> dict[str, str]:
    """Build the placeholder values for the templates.

    Args:
        report: The forecast.
        place: The resolved place.
        day: The resolved day.
        today: The date that counts as today.
        snow_first: Whether to report snow before rain.

    Returns:
        A mapping of placeholder name to text.
    """
    place_text = describe_place(place)
    day_text = describe_day(day, today)
    at_place = f"{preposition_for(place)} {place_text}"

    return {
        "place": place_text,
        "in_place": at_place,
        "in_place_cap": at_place[0].upper() + at_place[1:],
        "day": day_text,
        "day_cap": day_text[0].upper() + day_text[1:],
        "tmin": format_temperature(report.temperature_min),
        "tmax": format_temperature(report.temperature_max),
        "sky": SKY_PHRASES.get(report.sky, report.sky),
        "cloud": f"{report.cloud_cover:.0f}",
        "wind": f"{report.wind_speed_max:.0f} km/h",
        "gusts": f"{report.wind_gusts_max:.0f} km/h",
        "precipitation": describe_precipitation(report, snow_first),
    }


def generate_answer(
    intent: str,
    report: WeatherReport,
    place: Place,
    day: DaySlot,
    today: date,
    message: str = "",
    rng: random.Random | None = None,
    include_hint: bool = True,
    brief_offline_note: bool = False,
) -> str:
    """Write the answer for one weather question.

    Args:
        intent: The weather intent to answer.
        report: The forecast.
        place: The resolved place.
        day: The resolved day.
        today: The date that counts as today, for "today"/"tomorrow" wording.
        message: The user's own message, used only to decide whether snow is
            reported before rain.
        rng: Random source for picking a phrasing. Pass a seeded
            ``random.Random`` to make the output deterministic; the tests do.
        include_hint: Whether to append an outdoor hint.
        brief_offline_note: Use the short form of the offline disclosure,
            for answers after the first one in a conversation.

    Returns:
        One or two sentences, plus the offline note when the numbers are demo
        data.
    """
    rng = rng or random.Random()
    snow_first = mentions_snow(message)

    templates = TEMPLATES.get(intent, TEMPLATES[ASK_OVERVIEW])
    values = _template_values(report, place, day, today, snow_first)
    sentences = [rng.choice(templates).format(**values)]

    if include_hint:
        hints = build_hints(
            report, snow_first, stated=VALUES_STATED_BY_INTENT.get(intent, frozenset())
        )
        if hints:
            # One hint only. Two is a lecture, and the list is ordered so the
            # most consequential one is the one that survives.
            sentences.append(hints[0])

    if report.is_offline:
        sentences.append(OFFLINE_NOTE_BRIEF if brief_offline_note else OFFLINE_NOTE)

    return " ".join(sentences)


def fuzzy_confirmation(place: Place) -> str:
    """Say which place was assumed, when the name was not an exact match.

    ``Place.fuzzy`` is set when the name needed typo tolerance ("Innsbruk") or
    was a shortened compound ("Bischling" for Bischlinghöhe). Answering those
    silently would mean confidently reporting a place the user never typed, so
    the assumption is stated.

    Args:
        place: The resolved place.

    Returns:
        A short prefix sentence.
    """
    return f"I took that as {place.label}."
