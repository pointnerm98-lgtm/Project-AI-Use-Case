"""Slot extraction for PeakCast: which place, and which day.

The intent classifier answers *what* the user asked. This module answers *where*
and *when*, which is the other half of the NLU box in the architecture. Both
slots are extracted by **rules plus a gazetteer lookup**, not by a learned
model, and that is a deliberate decision rather than a shortcut:

* The day vocabulary is a **closed set** ("today", "tomorrow", the seven
  weekday names, "this afternoon", "in 3 days"). A rule covers it exactly and
  can be read and audited; a classifier would only approximate it, would need
  its own training data, and would occasionally resolve "Friday" to the wrong
  date with no way to tell why.
* Place names come from an **authoritative list** (GeoNames Austria, see
  ``scripts/build_places.py``). The bot must return real coordinates and a real
  elevation to query a forecast, so the place slot is a lookup problem, not a
  prediction problem. A model that "predicts" coordinates would be strictly
  worse than reading them from a gazetteer.

This is the same argument as the avalanche guard in ``peakcast.safety_guard``,
applied to a non-safety problem: where the answer must be exact and the input
space is enumerable, a rule beats a learned model.

What this module does *not* do
------------------------------
When no place is found it returns ``None``. Falling back to the place the user
mentioned earlier in the conversation is the **dialogue manager's** job - it is
the component that owns conversation state, and mixing that memory in here
would make this module untestable in isolation.
"""

from __future__ import annotations

import csv
import difflib
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

# --- Paths -----------------------------------------------------------------
# Defined here rather than imported from intent_classifier so that
# scripts/build_places.py can import the output path without pulling in
# scikit-learn.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"

# The compact gazetteer, built by scripts/build_places.py and committed.
PLACES_CSV: Path = DATA_DIR / "places_at.csv"

# Hand-curated short names that GeoNames does not carry, checked *before* the
# gazetteer. See ``load_aliases`` for why this file exists at all.
ALIASES_CSV: Path = DATA_DIR / "place_aliases.csv"

# --- Time ------------------------------------------------------------------
# All day words resolve against Austrian local time. Using the machine's local
# timezone instead would silently shift "today" for anyone running the bot
# outside Austria - including the tutor reviewing this project.
VIENNA: ZoneInfo = ZoneInfo("Europe/Vienna")

# Open-Meteo's free forecast covers 7 days including today, so day 0 to day 6.
# Anything beyond that is flagged rather than silently clamped: telling the user
# "I can only forecast 7 days ahead" is a correct answer, while answering for
# the wrong day is not.
FORECAST_DAYS: int = 7
MAX_DAY_OFFSET: int = FORECAST_DAYS - 1

# --- Parts of day ----------------------------------------------------------
# Kept as a separate flag on the day slot rather than as separate "days",
# because the weather service fetches a whole day and the response generator
# narrows it down to the right hours.
PART_MORNING: str = "morning"
PART_AFTERNOON: str = "afternoon"
PART_EVENING: str = "evening"
PART_NIGHT: str = "night"

WEEKDAY_NAMES: dict[str, int] = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
    # Common short forms. A user typing on a phone at a trailhead abbreviates.
    "mon": 0,
    "tue": 1,
    "tues": 1,
    "wed": 2,
    "weds": 2,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}

# Prepositions that mark the following word as a place. Used to rescue place
# names that are also ordinary English words (see AMBIGUOUS_ENGLISH_WORDS).
#
# "to" is deliberately **not** in this list, although "I'm going to Innsbruck"
# is perfectly good English. It was in the first version and caused exactly the
# false positive this mechanism exists to prevent:
#
#   "is it going to rain in Wien"  ->  Rain, Kärnten (pop 1 081)
#
# because "to rain" looks like "to <place>". Austria has 24 gazetteer entries
# named "Rain", and "going to rain" is one of the most common phrasings a
# weather bot sees, so the trade is not close. "from", "over" and "above" are
# left out for the same reason: they are rare before a place name in this
# domain and each one adds another way for a verb to be read as a location.
PLACE_PREPOSITIONS: frozenset[str] = frozenset(
    {"in", "at", "on", "near", "for", "around"}
)

# --- Ambiguous names -------------------------------------------------------
# The single most dangerous failure mode of a gazetteer matcher: Austria has
# real places whose names are ordinary English words, so a naive longest-match
# turns "is it going to rain" into a forecast for the village of Rain.
#
# Measured against the built gazetteer, the collisions are not hypothetical:
#
#   "Rain"    13 populated places, the largest with 1 081 inhabitants
#   "Hard"     a town in Vorarlberg with 13 495 inhabitants
#   "Going"    Going am Wilden Kaiser, Tirol (358)
#   "See"      7 populated places, the largest with 1 265
#   "Still", "First", "Alm", "Quick", "Not", "Fell", "Bad", "Astein"
#
# Dropping these names is not an option - "Hard" is a real town of 13 000
# people and someone there will ask for its weather. So they are kept, but
# matched only **with evidence**: either a preposition directly in front
# ("rain in Rain") or an initial capital in the user's own text ("Rain").
# Everything else in the gazetteer matches without that requirement.
#
# The list below is a curated set of ordinary English words, deliberately
# including the weather and outdoor vocabulary this bot sees most. It is the
# single source of truth: scripts/build_places.py imports it to report which
# gazetteer names it actually hits, so the collision list in the comment above
# can be re-measured instead of trusted.
AMBIGUOUS_ENGLISH_WORDS: frozenset[str] = frozenset(
    {
        # --- function words
        "a", "an", "the", "and", "or", "but", "if", "so", "as", "at", "by",
        "for", "from", "in", "into", "of", "off", "on", "out", "over", "to",
        "up", "with", "am", "is", "are", "be", "been", "do", "does", "did",
        "have", "has", "had", "will", "would", "can", "could", "should",
        "it", "its", "this", "that", "these", "those", "there", "here",
        "not", "no", "yes", "all", "any", "some", "both", "each", "more",
        "most", "much", "many", "few", "less", "least", "very", "just",
        "only", "also", "even", "still", "then", "than", "when", "where",
        "what", "which", "who", "how", "why", "now", "my", "me", "i",
        # --- weather and sky
        "rain", "snow", "sun", "wind", "gust", "cloud", "clouds", "fog",
        "storm", "hail", "ice", "frost", "sleet", "shower", "showers",
        "warm", "cold", "hot", "cool", "mild", "wet", "dry", "clear",
        "sunny", "cloudy", "rainy", "windy", "foggy", "snowy", "fresh",
        "weather", "forecast", "temperature", "degrees",
        # --- time
        "today", "tomorrow", "tonight", "day", "days", "night", "morning",
        "afternoon", "evening", "week", "weekend", "monday", "tuesday",
        "wednesday", "thursday", "friday", "saturday", "sunday", "next",
        "last", "early", "late", "soon",
        # --- terrain and outdoor
        "mountain", "peak", "summit", "hill", "pass", "valley", "lake",
        "river", "glacier", "trail", "hut", "alm", "top", "high", "low",
        "see", "view", "north", "south", "east", "west", "side", "face",
        # --- ordinary verbs, adjectives and nouns that turn up in questions
        "going", "go", "get", "got", "give", "take", "come", "coming",
        "need", "want", "know", "tell", "say", "see", "look", "think",
        "good", "bad", "nice", "fine", "great", "best", "worse", "worst",
        "hard", "easy", "first", "second", "third", "last", "new", "old",
        "big", "small", "long", "short", "near", "far", "open", "close",
        "closed", "safe", "quick", "fast", "slow", "well", "right", "left",
        "fell", "fall", "rise", "drop", "stay", "wear", "bring", "pack",
        "plan", "hike", "climb", "ski", "walk", "run", "ride", "fly",
        "jacket", "boots", "gear", "map", "time", "thing", "place", "people",
        "please", "thanks", "hello", "bye",
        # --- words English always capitalises
        # These need their own group, because for them the capitalisation
        # evidence is worthless: "German" is capitalised in every sentence it
        # appears in, so an initial capital says nothing about whether it is
        # being used as a place. Measured on the intent data, "can you speak
        # German" matched the village of Germanns (a prefix hit on "German"
        # with an initial capital), which is exactly this failure.
        "german", "austrian", "english", "italian", "french", "swiss",
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
    }
)

# --- Fuzzy matching --------------------------------------------------------
# Typo tolerance for place names ("Innsbruk" -> "Innsbruck"). Kept
# conservative on purpose: a wrong place silently produces a forecast for
# somewhere the user never asked about, which is worse than admitting the name
# was not recognised. Three guards:
#
#   1. a high similarity cut-off,
#   2. a minimum length, because short strings are similar to each other by
#      accident ("See"/"Set", "Alm"/"Alp"),
#   3. only candidates that already look like a place - capitalised in the
#      user's text, or preceded by a preposition.
FUZZY_CUTOFF: float = 0.86
# The stricter cut-off for a token that has neither a capital nor a preposition
# in front (step 11, see EVIDENCE_FREE_MIN_LENGTH).
FUZZY_CUTOFF_NO_EVIDENCE: float = 0.90
FUZZY_MIN_LENGTH: int = 5

# --- Partial names ---------------------------------------------------------
# Users shorten compound mountain names, and GeoNames stores the long form:
#
#   user says      gazetteer has
#   Krippenstein   Hoher Krippenstein
#   Bischling      Bischlinghöhe
#   Sonnwendjoch   Hinteres Sonnwendjoch
#
# This is a different problem from a typo and needs a different mechanism.
# Measured on the 400 intent sentences, fuzzy matching handled these badly -
# it found "Bichling" for "Bischling" and "Sonnenjoch" for "Sonnwendjoch",
# i.e. *different real places*, which is the worst kind of error because the
# answer looks perfectly plausible. Containment is tried first and gets all
# three right.
#
# Three guards keep it from over-reaching: a minimum length, the requirement
# that the typed name make up at least ``CONTAINMENT_MIN_COVERAGE`` of the
# full name, and the same place evidence the fuzzy matcher needs. Among
# several hits the **shortest** full name wins - the name that adds least to
# what the user actually typed.
CONTAINMENT_MIN_LENGTH: int = 6
CONTAINMENT_MIN_COVERAGE: float = 0.5

# --- Evidence for long names (step 11) --------------------------------------
# The capital-or-preposition evidence was introduced for the short names that
# are also English words ("Going", "Rain", "Hard"). It was applied to the
# typo-tolerant and partial stages as well, which meant that a phone user
# typing "snow obertauren tomorrow" - lower case, no "in" - got nothing, even
# though no English word is anywhere near "obertauren". From this length on, a
# token is long enough to be distinctive and needs no such evidence; shorter
# tokens keep the old rule. Day words and their typos are excluded explicitly
# ("tomoorow" must never become a village).
EVIDENCE_FREE_MIN_LENGTH: int = 7

# Long ordinary words that the partial and typo-tolerant stages would otherwise
# compare with the gazetteer once they no longer need evidence. Found by the
# lower-case sweep over the intent data (DEVLOG step 11); domain words, not a
# general dictionary.
COMMON_LONG_WORDS: frozenset[str] = frozenset(
    {
        "information", "conditions", "condition", "visibility", "freezing",
        "thermals", "touring", "backcountry", "paragliding", "climbing",
        "skitouring", "chances", "humidity", "planning", "thunderstorm",
        "thunderstorms", "precipitation", "snowfall", "sunshine", "overcast",
        "outside", "anything", "something", "everything", "nothing",
        "probably", "actually", "expecting", "expected", "looking", "raining",
        "snowing", "sunrise", "sunset", "minutes", "between", "another",
        "possible", "perfect", "awesome", "goodbye", "wether", "whether",
        "temperatur", "favourite", "favorite",
    }
)

# "st.johann" (no space after the dot) is how the tester typed it, and the
# tokeniser kept it as one token "stjohann", which matches no multi-word name.
# A space is inserted after "st." / "sankt." before tokenising.
_SAINT_DOT_RE = re.compile(r"\b(st|sankt)\.(?=[^\W\d_])", flags=re.IGNORECASE)

# Longest gazetteer name in tokens ("Maria Alm am Steinernen Meer" is 5).
# Bounds the n-gram scan so lookup stays linear in message length.
MAX_NAME_TOKENS: int = 6

# Feature-code preference for breaking ties between same-named places that
# have no population to compare (two peaks called the same thing). Populated
# places first, because a user naming a bare word more often means the village.
_FEATURE_CODE_RANK: dict[str, int] = {
    code: rank
    for rank, code in enumerate(
        [
            "PPLC", "PPLA", "PPLA2", "PPLA3", "PPLA4", "PPL", "PPLL", "PPLX",
            "MT", "PK", "MTS", "PASS", "GLCR", "LK", "SDL", "RDGE", "HLL",
            "VAL", "CRQ", "SLP", "HUT",
        ]
    )
}


@dataclass(frozen=True)
class Place:
    """One place from the gazetteer.

    Attributes:
        name: Canonical GeoNames name. Used for **resolution only** - looking
            the place up, comparing rows, keying the cache. It is not
            necessarily what a user should be shown; see ``display_name``.
        display_name: The name to put in an answer. GeoNames' canonical name
            is sometimes not the one an Austrian user knows the place by:
            the Mittagskogel is filed under its Slovene name *Kepa*, and the
            Mölltaler Gletscher under *Wurtenkees*. Answering "Kepa" would
            read as a different mountain. So the display name is the gazetteer
            spelling that matches what the user actually typed, preferring the
            German form; it falls back to ``name`` when the user's wording was
            a typo, an abbreviation or an alias (see ``choose_display_name``).
        state: Austrian federal state, so the answer can say which of several
            same-named places was used. Empty for features that do not lie in
            one single state (mountain ranges and border peaks such as the
            Hohe Tauern or the Raxalpe), where GeoNames supplies no code.
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        elevation: Metres above sea level, or ``None`` if unknown.
        population: Inhabitants; 0 for terrain features.
        feature_code: GeoNames feature code (``PPL``, ``MT``, ``PK``, ...).
        matched_text: The substring of the user message that matched.
        fuzzy: ``True`` if the match needed typo tolerance or a partial name
            rather than being exact. The dialogue manager uses this to confirm
            back ("I assumed you meant Innsbruck").
        ambiguous: ``True`` if more than one distinct place shared the name the
            user typed, so the answer should name the federal state to show
            which one was used.
    """

    name: str
    state: str
    lat: float
    lon: float
    elevation: int | None
    population: int
    feature_code: str
    matched_text: str = ""
    fuzzy: bool = False
    display_name: str = ""
    ambiguous: bool = False

    @property
    def shown(self) -> str:
        """Return the name to show the user.

        Falls back to the canonical name, so a ``Place`` built by hand (as the
        tests do) still renders sensibly without setting ``display_name``.
        """
        return self.display_name or self.name

    @property
    def label(self) -> str:
        """Return ``"Name, State"``, or just the name if the state is unknown.

        Used where the place has to be fully identified - logs, the CLI, the
        ask-back. The response generator names the state only when the name
        was ambiguous, which is what ``ambiguous`` is for.
        """
        return f"{self.shown}, {self.state}" if self.state else self.shown


@dataclass(frozen=True)
class DaySlot:
    """A resolved day reference.

    Attributes:
        target_date: The absolute date the user meant.
        offset_days: Days from today; 0 is today.
        part_of_day: ``"morning"``, ``"afternoon"``, ``"evening"``,
            ``"night"`` or ``None`` for the whole day.
        matched_text: The phrase that produced this, or ``""`` for the default.
        is_default: ``True`` if no day word was found and today was assumed.
        in_range: ``False`` if the date lies outside the forecast horizon.
        part_only: ``True`` if the phrase named a part of the day **without
            saying which day** - "in the afternoon", as opposed to "this
            afternoon" or "tomorrow afternoon", which are explicit about the
            day and leave this ``False``.

            The distinction exists for one job, in the dialogue manager: a
            follow-up like *"and in the afternoon?"* should inherit the day
            from the previous question, while *"this afternoon"* means today
            whatever came before. Resolving the bare form to today would
            silently answer about the wrong day halfway through a conversation
            about tomorrow.
        named_date: ``True`` if the day came from a weekday name or a calendar
            date ("next tuesday", "24.11."). Such answers spell the date out
            ("on Tuesday 24 November"), so that a misreading of which Tuesday
            was meant is visible to the user (step 11).
        span_days: How many consecutive days this slot covers, starting at
            ``target_date``. 1 for an ordinary day; the forecast horizon for a
            range such as "this week" or "the next days".
    """

    target_date: date
    offset_days: int
    part_of_day: str | None = None
    matched_text: str = ""
    is_default: bool = False
    in_range: bool = True
    part_only: bool = False
    named_date: bool = False
    span_days: int = 1

    @property
    def is_past(self) -> bool:
        """Return ``True`` for a day before today ("yesterday", "last week")."""
        return self.offset_days < 0

    @property
    def is_range(self) -> bool:
        """Return ``True`` for a multi-day range ("this week")."""
        return self.span_days > 1


@dataclass(frozen=True)
class Slots:
    """Everything the NLU extracted from one message besides the intent.

    Attributes:
        place: The place found, or ``None``.
        day: The resolved day; never ``None``, because today is the default.
            When the message named several days, this is the first.
        extra_days: The further days the message named, in order ("and on
            thursday? and friday?", or the Sunday of "this weekend"). Empty in
            the ordinary one-day case.
    """

    place: Place | None
    day: DaySlot
    extra_days: tuple[DaySlot, ...] = ()

    @property
    def all_days(self) -> tuple[DaySlot, ...]:
        """Return every day named, the first one included."""
        return (self.day, *self.extra_days)


# --- Text normalisation ----------------------------------------------------
# Applied identically to gazetteer names and to the user message, so the two
# sides always meet in the middle.
#
# The umlaut handling is the part that matters. A user may type any of
# "Sölden", "Soelden" or "Solden", and GeoNames itself supplies "Sölden" and
# the ASCII form "Solden". Mapping the umlaut to the bare vowel *and* the
# "ae"/"oe"/"ue" digraph to the same bare vowel collapses all three onto one
# key. The digraph rule occasionally merges unrelated spellings (a genuine
# "ue" inside a name is folded too), but since both sides are normalised the
# same way, the only cost is a rare extra candidate - never a missed match.
_UMLAUT_MAP = str.maketrans({"ä": "a", "ö": "o", "ü": "u", "é": "e", "è": "e", "à": "a"})
_DIGRAPH_RE = re.compile(r"(?<=[a-z])(ae|oe|ue)")


def normalise_name(text: str) -> str:
    """Normalise a place name or message fragment for matching.

    Steps: lower-case, "ß" to "ss", umlauts to bare vowels, "ae"/"oe"/"ue"
    digraphs to bare vowels, "Sankt" and "St." unified to "st", punctuation to
    spaces, whitespace collapsed.

    Args:
        text: Raw name or fragment.

    Returns:
        The normalised string.

    Examples:
        >>> normalise_name("Sölden")
        'solden'
        >>> normalise_name("Soelden")
        'solden'
        >>> normalise_name("St. Johann im Pongau")
        'st johann im pongau'
        >>> normalise_name("Sankt Johann im Pongau")
        'st johann im pongau'
    """
    lowered = text.lower().replace("ß", "ss")
    lowered = lowered.translate(_UMLAUT_MAP)
    lowered = _DIGRAPH_RE.sub(lambda m: m.group(0)[0], lowered)

    # Punctuation to spaces, so "St." and "Zell-am-See" tokenise cleanly.
    cleaned = "".join(ch if ch.isalnum() else " " for ch in lowered)
    tokens = cleaned.split()

    # "Sankt Johann" and "St. Johann" are the same place; GeoNames uses both.
    tokens = ["st" if token == "sankt" else token for token in tokens]
    return " ".join(tokens)


def _collapse(normalised: str) -> str:
    """Remove the spaces from an already-normalised name.

    GeoNames is inconsistent about compound names: the pass everybody writes
    as "Turracher Höhe" is stored as the single word "Turracherhöhe". Indexing
    a space-free form as well makes both spellings find it.

    Args:
        normalised: Output of ``normalise_name``.

    Returns:
        The same string without spaces.
    """
    return normalised.replace(" ", "")


# --- Gazetteer -------------------------------------------------------------


@dataclass
class PlaceIndex:
    """Lookup structures over the gazetteer.

    Attributes:
        by_name: Normalised name -> candidate places.
        by_collapsed: Space-free normalised name -> candidate places.
        by_alias: Normalised curated alias -> the one place it names.
        display: ``(normalised key, canonical name)`` -> the spelling to show
            the user for that key. This is what lets "Mittagskogel" be
            answered as *Mittagskogel* rather than as GeoNames' canonical
            *Kepa*: the pair is keyed by what the user typed, so the same row
            can be shown under whichever of its names was asked for.
        places: Every place, for diagnostics.
    """

    by_name: dict[str, list[Place]]
    by_collapsed: dict[str, list[Place]]
    by_alias: dict[str, Place]
    display: dict[tuple[str, str], str]
    places: list[Place]

    def best(self, candidates: list[Place]) -> Place:
        """Choose between several places that share a name.

        Most populous wins: if someone says "Zell am See" they almost
        certainly mean the town of 7 600 rather than the hamlet. Where there is
        no population to compare (two peaks of the same name), fall back to the
        feature-code preference and then to the higher elevation, which is a
        reasonable proxy for the better-known summit.

        Args:
            candidates: Places sharing a name.

        Returns:
            The chosen place.
        """
        return max(
            candidates,
            key=lambda p: (
                p.population,
                -_FEATURE_CODE_RANK.get(p.feature_code, 99),
                p.elevation or 0,
            ),
        )


def load_places(csv_path: Path = PLACES_CSV) -> list[Place]:
    """Read the compact gazetteer.

    Args:
        csv_path: Path to ``places_at.csv``.

    Returns:
        Every place in the file.

    Raises:
        FileNotFoundError: If the file is missing, with the command to build it.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No place list at {csv_path}.\n"
            "Build it first:  python scripts/build_places.py"
        )

    places: list[Place] = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            elevation_raw = row["elevation"].strip()
            places.append(
                Place(
                    name=row["name"],
                    state=row["state"],
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    elevation=int(elevation_raw) if elevation_raw else None,
                    population=int(row["population"] or 0),
                    feature_code=row["feature_code"],
                )
            )
    return places


def choose_display_name(spellings: list[str], canonical: str) -> str:
    """Pick the spelling to show the user, among those the user could have meant.

    All candidates passed in normalise to the same key, i.e. they are the
    different ways of writing the one name the user typed. The order of
    preference:

    1. **The canonical name**, if it is one of them. "Soelden" and "Sölden"
       both normalise to ``solden``, and GeoNames' canonical spelling
       *Sölden* is the right one to print.
    2. **A spelling with diacritics**, otherwise. This is what rescues the
       cases the whole mechanism exists for: the Mittagskogel is filed under
       *Kepa*, so the canonical name does not normalise to what the user
       typed, and among the remaining spellings the accented German form is
       the one an Austrian user recognises.
    3. The longest remaining spelling, as a stable tie-break - longer usually
       means the fuller, less abbreviated form.

    Args:
        spellings: Candidate spellings, all normalising to the same key.
        canonical: The row's canonical GeoNames name.

    Returns:
        The spelling to display.
    """
    if canonical in spellings:
        return canonical

    accented = [s for s in spellings if any(ord(ch) > 127 for ch in s)]
    pool = accented or spellings
    return max(pool, key=lambda s: (len(s), s))


def load_aliases(csv_path: Path = ALIASES_CSV) -> list[dict[str, str]]:
    """Read the hand-curated alias list.

    Why a hand-written file is justified here, when the rest of the place data
    is taken from an authoritative source: GeoNames stores official names, and
    the names people *say* are sometimes shorter. The gap is small, closed and
    not learnable from the dump:

        "the Rax"        -> Raxalpe        (and "Rax" is a village of 615 in
                                            Burgenland, which wins on
                                            population)
        "the Glockner"   -> Großglockner   (otherwise the mountain Glöcknerin)
        "the Schmitten"  -> Schmittenhöhe  (otherwise a village 1 200 m lower)
        "the Sonnblick"  -> Hoher Sonnblick (otherwise the Stubacher
                                            Sonnblick, a different mountain)
        "Emberger Alm"   -> Emberger Alpe  (GeoNames spells it "Alpe")

    Each row is an alias plus the gazetteer name and federal state it points
    at, so the alias can never introduce coordinates of its own - it only
    selects an existing, verified row. Every entry carries a one-line reason,
    because an undocumented alias list is indistinguishable from a list of
    arbitrary preferences.

    Args:
        csv_path: Path to ``place_aliases.csv``.

    Returns:
        One dict per row. An empty list if the file is absent, so the matcher
        degrades to plain gazetteer behaviour rather than failing.
    """
    if not csv_path.exists():
        return []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=1)
def get_place_index(csv_path: Path = PLACES_CSV) -> PlaceIndex:
    """Build (and cache) the place lookup index.

    Cached because the gazetteer has tens of thousands of rows and the
    Streamlit app would otherwise rebuild it on every keystroke.

    Args:
        csv_path: Path to ``places_at.csv``.

    Returns:
        The index.
    """
    by_name: dict[str, list[Place]] = {}
    by_collapsed: dict[str, list[Place]] = {}
    display: dict[tuple[str, str], str] = {}
    places = load_places(csv_path)

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    for place, row in zip(places, rows):
        # Index the canonical name, the ASCII form and every kept alternate.
        spellings = {place.name, row["ascii_name"]}
        spellings.update(a for a in row["alt_names"].split("|") if a)

        # Group this row's spellings by the key they normalise to, so that the
        # display name can be chosen per key: the same row is shown as
        # "Mittagskogel" or as "Kepa" depending on which name was asked for.
        by_key: dict[str, list[str]] = {}
        for spelling in spellings:
            key = normalise_name(spelling)
            if key:
                by_key.setdefault(key, []).append(spelling)

        for key, key_spellings in by_key.items():
            by_name.setdefault(key, []).append(place)
            display[(key, place.name)] = choose_display_name(
                key_spellings, place.name
            )
            collapsed = _collapse(key)
            # Only index the collapsed form when it differs, to keep the
            # second dict small.
            if collapsed != key:
                by_collapsed.setdefault(collapsed, []).append(place)
                display.setdefault(
                    (collapsed, place.name), display[(key, place.name)]
                )

    # --- Resolve the curated aliases against the gazetteer just built.
    by_alias: dict[str, Place] = {}
    for row in load_aliases():
        target_name = row["geonames_name"].strip()
        target_state = row["state"].strip()
        matches = [
            place
            for place in by_name.get(normalise_name(target_name), [])
            if place.name == target_name and place.state == target_state
        ]
        if not matches:
            # Fail loudly. A typo in the alias file would otherwise silently do
            # nothing, and the bot would keep answering for the wrong place
            # with no sign that the alias was meant to fix it.
            raise ValueError(
                f"Alias {row['alias']!r} points at "
                f"{target_name!r} ({target_state or 'no state'}), "
                f"which is not in {csv_path.name}"
            )
        by_alias[normalise_name(row["alias"])] = matches[0]

    return PlaceIndex(
        by_name=by_name,
        by_collapsed=by_collapsed,
        by_alias=by_alias,
        display=display,
        places=places,
    )


def _resolved(
    index: PlaceIndex,
    candidates: list[Place],
    key: str,
    matched_text: str,
    fuzzy: bool = False,
) -> Place:
    """Build the ``Place`` a match returns.

    One construction point for all four match stages (alias, exact, partial,
    fuzzy), so they cannot drift apart on how the display name is chosen or
    when a name counts as ambiguous.

    Args:
        index: The place index, which holds the display-name table.
        candidates: The places sharing the matched name.
        key: The normalised key that matched, used to look the display name up.
        matched_text: What the user actually typed.
        fuzzy: Whether the match needed typo tolerance or a partial name.

    Returns:
        The chosen place, carrying its display name and ambiguity flag.
    """
    chosen = index.best(candidates)

    # "Ambiguous" means several *distinct* places answer to this name, so the
    # answer should say which federal state it used. Rows that are duplicates
    # of one another (GeoNames lists some places twice) do not count.
    distinct = {(p.name, p.state, round(p.lat, 2), round(p.lon, 2)) for p in candidates}

    return Place(
        name=chosen.name,
        state=chosen.state,
        lat=chosen.lat,
        lon=chosen.lon,
        elevation=chosen.elevation,
        population=chosen.population,
        feature_code=chosen.feature_code,
        matched_text=matched_text,
        fuzzy=fuzzy,
        display_name=index.display.get((key, chosen.name), chosen.name),
        ambiguous=len(distinct) > 1,
    )


def _has_place_evidence(
    raw_tokens: list[str], normalised_tokens: list[str], start: int
) -> bool:
    """Check whether an ambiguous name is being used as a place here.

    Required for names that are also ordinary English words. Two accepted
    kinds of evidence:

    * a preposition immediately before it - "rain in Rain", "weather at See";
    * an initial capital in the user's own text - "Rain", "Hard", "Going".

    Args:
        raw_tokens: Tokens of the original message, capitalisation intact.
        normalised_tokens: Tokens of the normalised message.
        start: Index of the first token of the candidate name.

    Returns:
        ``True`` if the name may be treated as a place.
    """
    if start > 0 and normalised_tokens[start - 1] in PLACE_PREPOSITIONS:
        return True
    # The raw and normalised token lists are aligned by construction (see
    # find_place), so the same index refers to the same word.
    if start < len(raw_tokens) and raw_tokens[start][:1].isupper():
        return True
    return False


def _needs_no_evidence(token: str) -> bool:
    """Report whether a token is long and distinctive enough to match as is.

    Used by the partial and fuzzy stages; see ``EVIDENCE_FREE_MIN_LENGTH``.

    Args:
        token: A normalised token.

    Returns:
        ``True`` if the token may match without a capital or a preposition.
    """
    return (
        len(token) >= EVIDENCE_FREE_MIN_LENGTH
        and token not in AMBIGUOUS_ENGLISH_WORDS
        and correct_day_word(token) is None
        and token not in COMMON_LONG_WORDS
    )


def _tokenise_aligned(message: str) -> tuple[list[str], list[str]]:
    """Split a message into raw and normalised tokens of equal length.

    Capitalisation is one of the two pieces of evidence for ambiguous names, so
    the matcher needs the original spelling of each token alongside its
    normalised form. Normalising token by token (rather than normalising the
    whole string and splitting) is what keeps the two lists aligned.

    Args:
        message: Raw user message.

    Returns:
        ``(raw_tokens, normalised_tokens)``.
    """
    raw_tokens: list[str] = []
    normalised_tokens: list[str] = []

    message = _SAINT_DOT_RE.sub(lambda m: m.group(1) + ". ", message)
    for raw in re.findall(r"[^\W_]+(?:[.'][^\W_]+)*\.?", message, flags=re.UNICODE):
        normalised = normalise_name(raw)
        if not normalised:
            continue
        # normalise_name can split one raw token into two ("St." stays one,
        # but a token like "Zell-am-See" was already split by the regex).
        # Keep the pairing one-to-one by collapsing any internal space.
        raw_tokens.append(raw)
        normalised_tokens.append(normalised.replace(" ", ""))

    return raw_tokens, normalised_tokens


def find_place(message: str, csv_path: Path = PLACES_CSV) -> Place | None:
    """Find the place a message refers to.

    Strategy, in order:

    1. **Longest match wins.** The message is scanned left to right and at each
       position the longest run of tokens that is a known name is taken. This
       is what makes "Zell am See" resolve to the town rather than stopping at
       the hamlet "Zell", and "St. Johann im Pongau" to the full name.
    2. **Space-free fallback** for GeoNames' compound spellings
       ("Turracher Höhe" -> "Turracherhöhe").
    3. **Fuzzy match** for single-token typos, under the conservative guards
       described at ``FUZZY_CUTOFF``.

    Args:
        message: Raw user message.
        csv_path: Gazetteer to use; overridden in tests.

    Returns:
        The place, or ``None`` if nothing was recognised. Returning ``None``
        is the honest answer: restoring the place from conversation history is
        the dialogue manager's responsibility.
    """
    index = get_place_index(csv_path)
    raw_tokens, normalised_tokens = _tokenise_aligned(message)
    if not normalised_tokens:
        return None

    # --- Stage 0: the curated aliases, checked before the gazetteer.
    # They exist precisely because the gazetteer gets these wrong, so letting
    # it answer first would defeat the purpose. Longest alias wins, so
    # "the rax" is preferred over "rax".
    alias_hit = _find_place_by_alias(index, raw_tokens, normalised_tokens)
    if alias_hit is not None:
        return alias_hit

    # Collect every window that matches, then take the longest. Scanning
    # left to right and returning the first hit is NOT the same thing, and
    # the difference is a real bug that this ordering fixes:
    #
    #   "forecast for Sankt Anton am Arlberg"
    #
    # contains the two-token window "for st" (from "for" + "Sankt", which
    # normalises to "st"), and collapsing its spaces gives "forst" - the
    # village of Forst. Leftmost-first returned that and never looked at the
    # four-token "st anton am arlberg" further right. Longest-wins prefers the
    # more specific name, and the leftmost one only breaks a tie.
    # (length, -start, candidates, matched key); the key is carried along so
    # the display name can be looked up for the spelling the user actually used.
    best_match: tuple[int, int, list[Place], str] | None = None

    for start in range(len(normalised_tokens)):
        longest = min(MAX_NAME_TOKENS, len(normalised_tokens) - start)
        for length in range(longest, 0, -1):
            window = normalised_tokens[start : start + length]
            key = " ".join(window)

            candidates = index.by_name.get(key)

            # Space-free fallback, for GeoNames' compound spellings: the pass
            # everyone writes "Turracher Höhe" is stored as one word,
            # "Turracherhöhe". It is tried only when no token in the window is
            # an ordinary English word - otherwise the fallback is exactly what
            # fuses "for"+"st" into "Forst", because removing the space lets
            # two unrelated words become one name.
            if candidates is None and not any(
                token in AMBIGUOUS_ENGLISH_WORDS for token in window
            ):
                collapsed = _collapse(key)
                candidates = index.by_name.get(collapsed) or index.by_collapsed.get(
                    collapsed
                )

            if not candidates:
                continue

            # A single-token name that is also an English word needs evidence.
            if length == 1 and key in AMBIGUOUS_ENGLISH_WORDS:
                if not _has_place_evidence(raw_tokens, normalised_tokens, start):
                    continue

            if best_match is None or (length, -start) > (
                best_match[0],
                best_match[1],
            ):
                # Record the key that actually hit: the space-free fallback
                # matches under `collapsed`, not under `key`.
                matched_key = key if key in index.by_name else _collapse(key)
                best_match = (length, -start, candidates, matched_key)
            # No shorter window starting here can beat this one.
            break

    if best_match is None:
        return _find_place_partial(
            index, raw_tokens, normalised_tokens
        ) or _find_place_fuzzy(index, raw_tokens, normalised_tokens)

    length, negative_start, candidates, key = best_match
    start = -negative_start
    return _resolved(
        index, candidates, key, " ".join(raw_tokens[start : start + length])
    )


def _find_place_by_alias(
    index: PlaceIndex, raw_tokens: list[str], normalised_tokens: list[str]
) -> Place | None:
    """Match a hand-curated short name, before the gazetteer is consulted.

    Longest alias first, so ``"the rax"`` beats the single token ``"rax"``.
    The match is exact against the normalised alias, with no fuzzy or partial
    tolerance: these are deliberate, hand-checked mappings, and loosening them
    would reintroduce the guesswork they exist to remove.

    Args:
        index: The place index, carrying the resolved aliases.
        raw_tokens: Original tokens.
        normalised_tokens: Normalised tokens.

    Returns:
        The aliased place with ``matched_text`` set to what the user typed, or
        ``None``.
    """
    if not index.by_alias:
        return None

    longest_alias = max(len(alias.split()) for alias in index.by_alias)

    for length in range(min(longest_alias, len(normalised_tokens)), 0, -1):
        for start in range(len(normalised_tokens) - length + 1):
            key = " ".join(normalised_tokens[start : start + length])
            place = index.by_alias.get(key)
            if place is None:
                continue
            # The alias is resolved against the place's own canonical key, not
            # against the alias text: "the Glockner" is not a spelling of
            # anything, so the answer should say "Großglockner".
            return _resolved(
                index,
                [place],
                normalise_name(place.name),
                " ".join(raw_tokens[start : start + length]),
            )

    return None


def _find_place_partial(
    index: PlaceIndex, raw_tokens: list[str], normalised_tokens: list[str]
) -> Place | None:
    """Match a shortened compound name against the full gazetteer name.

    Handles "Krippenstein" for "Hoher Krippenstein" and "Bischling" for
    "Bischlinghöhe" - cases where the user typed a correct but incomplete name,
    which fuzzy matching resolves to a *different real place* rather than to
    the intended one.

    Args:
        index: The place index.
        raw_tokens: Original tokens.
        normalised_tokens: Normalised tokens.

    Returns:
        The matched place with ``fuzzy=True`` (it is still an assumption worth
        confirming back to the user), or ``None``.
    """
    for position, token in enumerate(normalised_tokens):
        if len(token) < CONTAINMENT_MIN_LENGTH:
            continue
        if token in AMBIGUOUS_ENGLISH_WORDS:
            continue
        if correct_day_word(token) is not None:
            continue
        if not (
            _needs_no_evidence(token)
            or _has_place_evidence(raw_tokens, normalised_tokens, position)
        ):
            continue

        # Word-boundary containment only: the token must be a whole word of
        # the gazetteer name, or the start of its first word ("Bischling" in
        # "Bischlinghöhe"). A match in the middle of a word would let
        # "stein" find half of Austria.
        hits = [
            key
            for key in index.by_name
            if (
                key.startswith(token)
                or key.endswith(" " + token)
                or (" " + token + " ") in key
            )
            and len(token) / len(key) >= CONTAINMENT_MIN_COVERAGE
        ]
        if not hits:
            continue

        shortest = min(hits, key=len)
        return _resolved(
            index,
            index.by_name[shortest],
            shortest,
            raw_tokens[position],
            fuzzy=True,
        )

    return None


def _find_place_fuzzy(
    index: PlaceIndex, raw_tokens: list[str], normalised_tokens: list[str]
) -> Place | None:
    """Try a typo-tolerant match after exact matching failed.

    Only tokens that already look like a place are considered - capitalised, or
    introduced by a preposition - and only above ``FUZZY_CUTOFF``. Without
    those guards the matcher would quietly turn ordinary words into villages,
    which is a worse failure than not recognising the name at all.

    Args:
        index: The place index.
        raw_tokens: Original tokens.
        normalised_tokens: Normalised tokens.

    Returns:
        The matched place with ``fuzzy=True``, or ``None``.
    """
    for position, token in enumerate(normalised_tokens):
        if len(token) < FUZZY_MIN_LENGTH:
            continue
        if token in AMBIGUOUS_ENGLISH_WORDS:
            continue
        if correct_day_word(token) is not None:
            continue
        has_evidence = _has_place_evidence(raw_tokens, normalised_tokens, position)
        if not (has_evidence or _needs_no_evidence(token)):
            continue

        # Without capital or preposition the similarity bar is higher (step
        # 11): "favourite" is 0.889 similar to "Favoriten" and was read as the
        # Vienna district; a real typo such as "obertauren" scores 0.90.
        cutoff = FUZZY_CUTOFF if has_evidence else FUZZY_CUTOFF_NO_EVIDENCE
        matches = difflib.get_close_matches(
            token, index.by_name.keys(), n=1, cutoff=cutoff
        )
        if not matches:
            continue

        return _resolved(
            index,
            index.by_name[matches[0]],
            matches[0],
            raw_tokens[position],
            fuzzy=True,
        )

    return None


# --- Day extraction --------------------------------------------------------
# Rewritten in step 11 after user test round 1, which found five distinct
# failures in the day slot (DEVLOG step 11):
#
#   "tomoorow", "tmorrow"            typos ignored, silently answered for today
#   "what about next tuesday?"       read as the Tuesday of the following week
#   "wht temperature yesterday ..."  a past day answered as today, without a word
#   "... this week in obertauern"    a range answered as today
#   "and on thursday? and friday?"   only the first day answered
#
# The structure that fixes the last one fixes several others too: instead of
# returning the *first pattern that matches*, every day mention in the message
# is collected with its position, overlaps are resolved in favour of the longer
# phrase, and the mentions are returned in the order the user wrote them.
#
# Each fixed phrase is (regex, offset_days, part_of_day, part_only). Negative
# offsets are past days: they are returned, flagged by ``DaySlot.is_past``, so
# the dialogue manager can say that PeakCast only forecasts from today on rather
# than quietly answering for today.
_DAY_PATTERNS: list[tuple[str, int, str | None, bool]] = [
    # --- explicit two- and three-word phrases
    (r"\b(?:the )?day after tomorrow\b", 2, None, False),
    (r"\b(?:the )?day before yesterday\b", -2, None, False),
    (r"\btomorrow morning\b", 1, PART_MORNING, False),
    (r"\btomorrow afternoon\b", 1, PART_AFTERNOON, False),
    (r"\btomorrow evening\b", 1, PART_EVENING, False),
    (r"\btomorrow night\b", 1, PART_NIGHT, False),
    (r"\bthis morning\b", 0, PART_MORNING, False),
    (r"\bthis afternoon\b", 0, PART_AFTERNOON, False),
    (r"\bthis evening\b", 0, PART_EVENING, False),
    (r"\bthis night\b", 0, PART_NIGHT, False),
    (r"\blast night\b", -1, PART_NIGHT, False),
    # "tonight" is today plus a night flag, not a separate day: a user asking
    # at 18:00 about "tonight" means the hours still ahead of them today.
    (r"\btonight\b", 0, PART_NIGHT, False),
    (r"\bovernight\b", 0, PART_NIGHT, False),
    # --- single words
    (r"\btomorrow\b", 1, None, False),
    (r"\btmrw\b", 1, None, False),
    (r"\btoday\b", 0, None, False),
    (r"\bright now\b", 0, None, False),
    (r"\bnow\b", 0, None, False),
    (r"\bat the moment\b", 0, None, False),
    (r"\bcurrently\b", 0, None, False),
    (r"\byesterday\b", -1, None, False),
    (r"\blast week\b", -7, None, False),
    # --- "in the <part>": a part of the day with no day named. These are the
    #     only part_only=True patterns, because they are the only ones that
    #     leave the day open. On their own they mean today; next to a named day
    #     ("friday in the afternoon") they attach to it; inside a conversation
    #     the dialogue manager reads them as the remembered day, which is what
    #     makes "and in the afternoon?" work as a follow-up.
    #
    #     Bare "morning"/"evening" are deliberately NOT patterns: "good
    #     morning" and "have a good evening" are greetings, and the intent
    #     classifier already has `greet`/`goodbye` for them.
    (r"\bin the morning\b", 0, PART_MORNING, True),
    (r"\bin the afternoon\b", 0, PART_AFTERNOON, True),
    (r"\bin the evening\b", 0, PART_EVENING, True),
]
_COMPILED_DAY_PATTERNS = [
    (re.compile(pattern), offset, part, part_only)
    for pattern, offset, part, part_only in _DAY_PATTERNS
]

# --- Ranges ----------------------------------------------------------------
# "this week", "the next days", "the coming days": the user wants an overview
# of the period, not today's numbers. A range is one DaySlot that starts today
# and spans the whole forecast horizon, and the response generator turns it
# into one short summary (totals and the notable day) rather than seven
# answers. The forecast horizon rather than "until Sunday" is used for "this
# week", because the tester who asked it on a Sunday clearly meant the coming
# days - the calendar week had one day left.
RANGE_SPAN_DAYS: int = FORECAST_DAYS
_RANGE_RE = re.compile(
    r"\b(?:"
    r"(?:this|the|coming|the coming) week"
    r"|rest of the week"
    r"|(?:the )?(?:next|coming|following) (?:few )?days"
    r"|(?:the )?(?:next|coming) couple of days"
    r"|(?:the )?few days"
    r")\b"
)

# "this weekend" is **two** days, Saturday and Sunday, and both are answered
# (step 11; until then only Saturday was). Said on a Saturday or Sunday,
# "this weekend" still means the *coming* Saturday, as before - on a Sunday
# that Sunday's own successor then lies one day beyond the horizon, and the
# answer says so rather than dropping it.
_WEEKEND_RE = re.compile(r"\b(this |next |the |last )?weekend\b")

# "in 3 days", "in a week".
_IN_N_DAYS_RE = re.compile(r"\bin (\d+|a|one|two|three|four|five|six|seven) (day|days|week|weeks)\b")
_NUMBER_WORDS: dict[str, int] = {
    "a": 1, "one": 1, "two": 2, "three": 3,
    "four": 4, "five": 5, "six": 6, "seven": 7,
}

# Short forms that are also ordinary words: "how much sun tomorrow" is not about
# Sunday, and "I sat" is not Saturday. They count as days only with a prefix
# ("on sun", "this sat"). Found by the step 11 slot sweep: once every day
# mention in a message was collected, "sun" started to win over "tomorrow" -
# and a bare "how much sun in Saalbach" had been read as Sunday all along.
PREFIX_ONLY_DAY_ABBREVIATIONS: frozenset[str] = frozenset({"sun", "sat"})

# "on Friday", "next Friday", "last Friday", "friday".
_WEEKDAY_RE = re.compile(
    r"\b(next |on |this |coming |last )?(" + "|".join(WEEKDAY_NAMES) + r")\b"
)

# --- Explicit calendar dates -----------------------------------------------
# Before step 11 the bot only understood dates through a weekday name; the
# tester's "the 24th of november" was silently answered for today. Accepted
# forms: "24 November", "24th of November", "November 24", "Nov 24th",
# "24.11." / "24.11.2026" (the German way of writing it), and "the 24th"
# (the next 24th from today).
#
# Month names are only ever matched next to a number, so "may" in "it may
# rain" and "march" as a verb can never become a date.
MONTH_NAMES: dict[str, int] = {
    "january": 1, "jan": 1, "januar": 1, "jaenner": 1, "janner": 1,
    "february": 2, "feb": 2, "februar": 2,
    "march": 3, "mar": 3, "marz": 3, "maerz": 3, "märz": 3,
    "april": 4, "apr": 4,
    "may": 5, "mai": 5,
    "june": 6, "jun": 6, "juni": 6,
    "july": 7, "jul": 7, "juli": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "oktober": 10, "okt": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12, "dezember": 12, "dez": 12,
}
_MONTH_ALT = "|".join(sorted(MONTH_NAMES, key=len, reverse=True))
_ORDINAL = r"(?:st|nd|rd|th)?"
_DATE_DAY_MONTH_RE = re.compile(
    rf"\b(?:the )?(\d{{1,2}}){_ORDINAL}(?: of)? ({_MONTH_ALT})\b"
)
_DATE_MONTH_DAY_RE = re.compile(
    rf"\b({_MONTH_ALT}) (?:the )?(\d{{1,2}}){_ORDINAL}\b"
)
# "24.11." needs the trailing dot (or a year), so a decimal such as "2.5" is
# never read as a date.
_DATE_NUMERIC_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4}|\d{2})?")
_DATE_ORDINAL_ONLY_RE = re.compile(r"\bthe (\d{1,2})(?:st|nd|rd|th)\b")

# A date more than this many days in the past is taken to mean next year:
# asked on 30 December, "5 January" is a week ahead, not a year ago. Anything
# closer stays in the past and is answered as a past day.
DATE_ROLLOVER_DAYS: int = 60

# At most this many days are answered from one message ("and on thursday? and
# friday?"). Three keeps the reply readable on a phone; a fourth day is
# dropped rather than turning the answer into a table.
MAX_DAYS_PER_MESSAGE: int = 3

# --- Day-word typos --------------------------------------------------------
# The day vocabulary is closed, which makes typo tolerance safe in a way it is
# not for place names. Measured on the user test: "tomoorow", "tmorrow",
# "tomorow", "tommorow", "satruday". Rules, all conservative:
#
#   * only words of DAY_FUZZY_MIN_LENGTH letters or more ("today" and "now"
#     are too short to tolerate typos in - "toady", "nowt" ...);
#   * one edit for six- and seven-letter words, two from eight letters on
#     (optimal string alignment, so a swapped pair is one edit);
#   * never a word that is itself an ordinary English word, a weather word
#     ("sunnyy" is two edits from "sunday") or a place in the gazetteer
#     ("Mondsee" must stay a lake);
#   * the correction must be unique - a word equally close to two day words is
#     left alone.
DAY_FUZZY_MIN_LENGTH: int = 6
DAY_FUZZY_LONG_WORD: int = 8
DAY_VOCABULARY: tuple[str, ...] = (
    "tomorrow", "yesterday", "tonight", "weekend",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
)
# Words close to a day word that must never be corrected into one.
DAY_FUZZY_PROTECTED: frozenset[str] = frozenset(
    {
        "sunny", "sunnyy", "sunnny", "sundae", "sunday", "summit", "summer",
        "windy", "monday", "mondays", "fridays", "tonite", "tomato", "tomatoes",
        "weekends", "weekly", "weekday", "weekdays", "saturn", "thursdays",
        "tuesdays", "wednesdays", "saturdays", "sundays", "yesterdays",
    }
)


def today_in_vienna() -> date:
    """Return the current date in Austrian local time.

    Separate function so tests can monkeypatch "now" without touching the
    system clock.

    Returns:
        Today's date in ``Europe/Vienna``.
    """
    return datetime.now(VIENNA).date()


def correct_day_word(word: str) -> str | None:
    """Return the day word a typo stands for, or ``None``.

    Args:
        word: One lower-case token.

    Returns:
        The canonical day word ("tomorrow", "saturday", ...) if ``word`` is
        that word or a conservative typo of it, otherwise ``None``.
    """
    if word in DAY_VOCABULARY:
        return word
    if (
        len(word) < DAY_FUZZY_MIN_LENGTH
        or not word.isalpha()
        or word in DAY_FUZZY_PROTECTED
        or word in AMBIGUOUS_ENGLISH_WORDS
    ):
        return None

    allowed = 2 if len(word) >= DAY_FUZZY_LONG_WORD else 1
    # Imported lazily: textmatch has no dependencies, but keeping the import
    # here makes the one place the day extractor uses it obvious.
    from peakcast.textmatch import osa_distance

    scored = sorted(
        (osa_distance(word, target), target) for target in DAY_VOCABULARY
    )
    best_distance, best_target = scored[0]
    if best_distance > allowed:
        return None
    if len(scored) > 1 and scored[1][0] == best_distance:
        return None  # equally close to two day words: do not guess

    # Last guard, checked only for words that would otherwise be corrected:
    # a real place name is never a typo.
    if normalise_name(word) in get_place_index().by_name:
        return None
    return best_target


def _correct_day_typos(text: str) -> str:
    """Replace day-word typos in an already lower-cased text by the real word.

    Args:
        text: Lower-cased message with punctuation (except dots) as spaces.

    Returns:
        The text with each typo replaced.
    """
    tokens = text.split(" ")
    corrected = []
    for token in tokens:
        core = token.rstrip(".")
        fixed = correct_day_word(core) if core else None
        corrected.append(fixed + token[len(core):] if fixed and fixed != core else token)
    return " ".join(corrected)


def _make_day(
    offset: int,
    part_of_day: str | None,
    matched_text: str,
    reference: date,
    part_only: bool = False,
    named_date: bool = False,
    span_days: int = 1,
) -> DaySlot:
    """Build a ``DaySlot`` and flag it if it is outside the forecast horizon.

    Args:
        offset: Days from the reference date; negative for a past day.
        part_of_day: Part-of-day flag or ``None``.
        matched_text: The phrase that produced this.
        reference: The date that counts as "today".
        part_only: Whether the phrase named a part of the day but not which
            day; see ``DaySlot.part_only``.
        named_date: Whether the day came from a weekday name or a calendar
            date, so the answer must spell the date out.
        span_days: Number of days covered, for ranges.

    Returns:
        The day slot, with ``in_range=False`` if the bot cannot forecast it.
    """
    return DaySlot(
        target_date=reference + timedelta(days=offset),
        offset_days=offset,
        part_of_day=part_of_day,
        matched_text=matched_text,
        in_range=0 <= offset <= MAX_DAY_OFFSET,
        part_only=part_only,
        named_date=named_date,
        span_days=span_days,
    )


def _resolve_calendar_date(day_number: int, month: int, reference: date, year: int | None = None) -> date | None:
    """Turn a day and month into a date, choosing the year sensibly.

    Args:
        day_number: Day of the month.
        month: Month number.
        reference: Today.
        year: An explicit year, if the user gave one.

    Returns:
        The date, or ``None`` if it does not exist (31 November).
    """
    try:
        if year is not None:
            return date(year, month, day_number)
        candidate = date(reference.year, month, day_number)
        if (reference - candidate).days > DATE_ROLLOVER_DAYS:
            candidate = date(reference.year + 1, month, day_number)
        return candidate
    except ValueError:
        return None


def _date_mentions(text: str, reference: date) -> list[tuple[int, int, DaySlot]]:
    """Find explicit calendar dates.

    Args:
        text: Normalised message (dots kept).
        reference: Today.

    Returns:
        ``(start, end, day)`` per date found.
    """
    found: list[tuple[int, int, DaySlot]] = []

    def add(match: re.Match, resolved: date | None) -> None:
        if resolved is None:
            return
        found.append(
            (
                match.start(),
                match.end(),
                _make_day(
                    (resolved - reference).days,
                    None,
                    match.group(0),
                    reference,
                    named_date=True,
                ),
            )
        )

    for match in _DATE_DAY_MONTH_RE.finditer(text):
        add(match, _resolve_calendar_date(int(match.group(1)), MONTH_NAMES[match.group(2)], reference))
    for match in _DATE_MONTH_DAY_RE.finditer(text):
        add(match, _resolve_calendar_date(int(match.group(2)), MONTH_NAMES[match.group(1)], reference))
    for match in _DATE_NUMERIC_RE.finditer(text):
        year_text = match.group(3)
        year = None
        if year_text:
            year = int(year_text) + (2000 if len(year_text) == 2 else 0)
        month = int(match.group(2))
        if 1 <= month <= 12:
            add(match, _resolve_calendar_date(int(match.group(1)), month, reference, year))
    for match in _DATE_ORDINAL_ONLY_RE.finditer(text):
        day_number = int(match.group(1))
        # The next such day of the month from today on.
        resolved = _resolve_calendar_date(day_number, reference.month, reference, reference.year)
        if resolved is not None and resolved < reference:
            next_month = reference.month % 12 + 1
            next_year = reference.year + (1 if next_month == 1 else 0)
            resolved = _resolve_calendar_date(day_number, next_month, reference, next_year)
        add(match, resolved)
    return found


def _day_mentions(text: str, reference: date) -> list[tuple[int, int, list[DaySlot]]]:
    """Collect every day mention in the text with its position.

    One mention can produce two days (the weekend), so each entry carries a
    list.

    Args:
        text: Normalised, typo-corrected message.
        reference: Today.

    Returns:
        ``(start, end, days)`` per mention, unordered and possibly overlapping.
    """
    mentions: list[tuple[int, int, list[DaySlot]]] = []

    for pattern, offset, part, part_only in _COMPILED_DAY_PATTERNS:
        for match in pattern.finditer(text):
            mentions.append(
                (
                    match.start(),
                    match.end(),
                    [_make_day(offset, part, match.group(0), reference, part_only=part_only)],
                )
            )

    for match in _RANGE_RE.finditer(text):
        mentions.append(
            (
                match.start(),
                match.end(),
                [_make_day(0, None, match.group(0), reference, span_days=RANGE_SPAN_DAYS)],
            )
        )

    for match in _IN_N_DAYS_RE.finditer(text):
        raw_number, unit = match.group(1), match.group(2)
        count = int(raw_number) if raw_number.isdigit() else _NUMBER_WORDS[raw_number]
        offset = count * 7 if unit.startswith("week") else count
        mentions.append(
            (match.start(), match.end(), [_make_day(offset, None, match.group(0), reference)])
        )

    for match in _WEEKEND_RE.finditer(text):
        prefix = (match.group(1) or "").strip()
        saturday = (5 - reference.weekday()) % 7
        if prefix == "next":
            saturday += 7
        elif prefix == "last":
            saturday = -((reference.weekday() - 5) % 7 or 7)
        mentions.append(
            (
                match.start(),
                match.end(),
                [
                    _make_day(saturday, None, match.group(0), reference, named_date=True),
                    _make_day(saturday + 1, None, match.group(0), reference, named_date=True),
                ],
            )
        )

    for match in _WEEKDAY_RE.finditer(text):
        prefix = (match.group(1) or "").strip()
        if match.group(2) in PREFIX_ONLY_DAY_ABBREVIATIONS and not prefix:
            continue
        weekday = WEEKDAY_NAMES[match.group(2)]
        if prefix == "last":
            offset = -((reference.weekday() - weekday) % 7 or 7)
        else:
            # "Friday" said on a Friday means today, not a week from now.
            #
            # "next Friday" means the same as "on Friday": the next upcoming
            # one, within seven days. Until step 11 it was read as the Friday
            # of the *following* week, which is one defensible reading of
            # English - and the wrong one for the tester, who asked "what about
            # next tuesday?" on a Sunday and got "Tuesday 1 December is too far
            # ahead". The answer now always names the date it used ("on
            # Tuesday 24 November"), so if a user did mean the week after, the
            # reading is visible and easy to correct.
            offset = (weekday - reference.weekday()) % 7
        mentions.append(
            (
                match.start(),
                match.end(),
                [_make_day(offset, None, match.group(0), reference, named_date=True)],
            )
        )

    for start, end, day in _date_mentions(text, reference):
        mentions.append((start, end, [day]))

    return mentions


def _normalise_day_text(message: str) -> str:
    """Lower-case, turn punctuation except dots into spaces, fix day typos.

    Dots are kept because "24.11." is a date; the patterns use word
    boundaries, so "tomorrow." still matches "tomorrow".

    Args:
        message: Raw user message.

    Returns:
        The text the day patterns run on.
    """
    text = "".join(ch if ch.isalnum() or ch == "." else " " for ch in message.lower())
    text = " ".join(text.split())
    return _correct_day_typos(text)


def find_days(message: str, reference: date | None = None) -> list[DaySlot]:
    """Find every day a message names, in the order the user wrote them.

    Up to ``MAX_DAYS_PER_MESSAGE`` days are returned. "this weekend" counts as
    two (Saturday and Sunday). A part of day that names no day of its own
    ("friday in the afternoon") is attached to the named day rather than
    returned as a separate today.

    Args:
        message: Raw user message.
        reference: The date to treat as today.

    Returns:
        The days, possibly empty.
    """
    reference = reference or today_in_vienna()
    text = _normalise_day_text(message)

    # Longest mention first, so "the day after tomorrow" beats "tomorrow" and
    # "tomorrow afternoon" beats both "tomorrow" and nothing; then keep every
    # mention that does not overlap one already kept.
    mentions = sorted(_day_mentions(text, reference), key=lambda m: (-(m[1] - m[0]), m[0]))
    kept: list[tuple[int, int, list[DaySlot]]] = []
    for start, end, days in mentions:
        if all(end <= k_start or start >= k_end for k_start, k_end, _ in kept):
            kept.append((start, end, days))
    kept.sort(key=lambda m: m[0])

    full_days: list[DaySlot] = []
    part_only_days: list[DaySlot] = []
    for _, _, days in kept:
        for day in days:
            (part_only_days if day.part_only else full_days).append(day)

    if not full_days:
        return part_only_days[:1]

    # "on friday in the afternoon": the part of day belongs to the named day.
    if part_only_days and len(full_days) == 1 and full_days[0].part_of_day is None:
        only = full_days[0]
        full_days = [
            DaySlot(
                target_date=only.target_date,
                offset_days=only.offset_days,
                part_of_day=part_only_days[0].part_of_day,
                matched_text=only.matched_text,
                in_range=only.in_range,
                named_date=only.named_date,
                span_days=only.span_days,
            )
        ]

    # The same day named twice ("tomorrow ... tomorrow") is one day.
    unique: list[DaySlot] = []
    for day in full_days:
        if not any(
            u.target_date == day.target_date
            and u.part_of_day == day.part_of_day
            and u.span_days == day.span_days
            for u in unique
        ):
            unique.append(day)
    return unique[:MAX_DAYS_PER_MESSAGE]


def default_day(reference: date | None = None) -> DaySlot:
    """Return the day assumed when the message names none: today.

    Args:
        reference: The date to treat as today.

    Returns:
        Today, flagged ``is_default``.
    """
    reference = reference or today_in_vienna()
    return DaySlot(
        target_date=reference,
        offset_days=0,
        part_of_day=None,
        matched_text="",
        is_default=True,
        in_range=True,
    )


def find_day(message: str, reference: date | None = None) -> DaySlot:
    """Resolve the (first) day a message refers to.

    Recognises ``today``, ``tonight``, ``this morning/afternoon/evening``,
    ``tomorrow``, ``the day after tomorrow``, the weekday names (with
    ``next``/``on``/``this``/``last``), ``this weekend``, ``in N days``,
    calendar dates, ranges ("this week") and past days ("yesterday"), with
    typo tolerance for the day words. See ``find_days`` for messages naming
    several days.

    Defaults to **today** when no day word is present, which matches what a
    user means by a bare "how windy is it on the Hochkönig".

    Dates beyond the forecast horizon, and past dates, are returned with
    ``in_range=False`` rather than being clamped, so the bot can say so
    instead of answering for the wrong day.

    Args:
        message: Raw user message.
        reference: The date to treat as today. Defaults to today in Vienna;
            passed explicitly by the tests so they do not depend on the clock.

    Returns:
        The resolved day; never ``None``.
    """
    days = find_days(message, reference=reference)
    return days[0] if days else default_day(reference)


def is_day_word(word: str) -> bool:
    """Report whether a single token is a day word or a typo of one.

    Used by the dialogue manager's follow-up test, which strips the day the
    user named and needs to strip a misspelled one as well.

    Args:
        word: One token, any case.

    Returns:
        ``True`` for "tomorrow", "tmorrow", "friday", "weekend", ...
    """
    return correct_day_word(word.lower()) is not None


def extract_slots(
    message: str, reference: date | None = None, csv_path: Path = PLACES_CSV
) -> Slots:
    """Extract both slots from one message.

    This is the function the dialogue manager calls after the safety guard has
    passed the message and the classifier has produced an intent.

    Args:
        message: Raw user message.
        reference: The date to treat as today.
        csv_path: Gazetteer to use; overridden in tests.

    Returns:
        The place (or ``None``), the first day named (today by default), and
        any further days the message named.
    """
    days = find_days(message, reference=reference)
    return Slots(
        place=find_place(message, csv_path=csv_path),
        day=days[0] if days else default_day(reference),
        extra_days=tuple(days[1:]),
    )
