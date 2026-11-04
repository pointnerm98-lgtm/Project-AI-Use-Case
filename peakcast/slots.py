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
        name: Canonical name, as the bot should echo it back.
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
        fuzzy: ``True`` if the match needed typo tolerance rather than being
            exact. The dialogue manager can use this to confirm back
            ("I assumed you meant Innsbruck").
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

    @property
    def label(self) -> str:
        """Return ``"Name, State"``, or just the name if the state is unknown.

        Used in the answer sentence so the user can tell which of several
        same-named places the forecast is for.
        """
        return f"{self.name}, {self.state}" if self.state else self.name


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
    """

    target_date: date
    offset_days: int
    part_of_day: str | None = None
    matched_text: str = ""
    is_default: bool = False
    in_range: bool = True


@dataclass(frozen=True)
class Slots:
    """Everything the NLU extracted from one message besides the intent.

    Attributes:
        place: The place found, or ``None``.
        day: The resolved day; never ``None``, because today is the default.
    """

    place: Place | None
    day: DaySlot


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
        places: Every place, for diagnostics.
    """

    by_name: dict[str, list[Place]]
    by_collapsed: dict[str, list[Place]]
    by_alias: dict[str, Place]
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
    places = load_places(csv_path)

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    for place, row in zip(places, rows):
        # Index the canonical name, the ASCII form and every kept alternate.
        spellings = {place.name, row["ascii_name"]}
        spellings.update(a for a in row["alt_names"].split("|") if a)

        for spelling in spellings:
            key = normalise_name(spelling)
            if not key:
                continue
            by_name.setdefault(key, []).append(place)
            collapsed = _collapse(key)
            # Only index the collapsed form when it differs, to keep the
            # second dict small.
            if collapsed != key:
                by_collapsed.setdefault(collapsed, []).append(place)

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
        places=places,
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
    best_match: tuple[int, int, list[Place]] | None = None  # (length, -start, places)

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
                best_match = (length, -start, candidates)
            # No shorter window starting here can beat this one.
            break

    if best_match is None:
        return _find_place_partial(
            index, raw_tokens, normalised_tokens
        ) or _find_place_fuzzy(index, raw_tokens, normalised_tokens)

    length, negative_start, candidates = best_match
    start = -negative_start
    chosen = index.best(candidates)
    return Place(
        name=chosen.name,
        state=chosen.state,
        lat=chosen.lat,
        lon=chosen.lon,
        elevation=chosen.elevation,
        population=chosen.population,
        feature_code=chosen.feature_code,
        matched_text=" ".join(raw_tokens[start : start + length]),
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
            return Place(
                name=place.name,
                state=place.state,
                lat=place.lat,
                lon=place.lon,
                elevation=place.elevation,
                population=place.population,
                feature_code=place.feature_code,
                # What the user typed, so the answer can use their wording.
                matched_text=" ".join(raw_tokens[start : start + length]),
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
        if not _has_place_evidence(raw_tokens, normalised_tokens, position):
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
        chosen = index.best(index.by_name[shortest])
        return Place(
            name=chosen.name,
            state=chosen.state,
            lat=chosen.lat,
            lon=chosen.lon,
            elevation=chosen.elevation,
            population=chosen.population,
            feature_code=chosen.feature_code,
            matched_text=raw_tokens[position],
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
        if not _has_place_evidence(raw_tokens, normalised_tokens, position):
            continue

        matches = difflib.get_close_matches(
            token, index.by_name.keys(), n=1, cutoff=FUZZY_CUTOFF
        )
        if not matches:
            continue

        candidates = index.by_name[matches[0]]
        chosen = index.best(candidates)
        return Place(
            name=chosen.name,
            state=chosen.state,
            lat=chosen.lat,
            lon=chosen.lon,
            elevation=chosen.elevation,
            population=chosen.population,
            feature_code=chosen.feature_code,
            matched_text=raw_tokens[position],
            fuzzy=True,
        )

    return None


# --- Day extraction --------------------------------------------------------
# Ordered longest-phrase-first. "day after tomorrow" has to be tested before
# "tomorrow", and "this afternoon" before "afternoon", or the shorter phrase
# would swallow the longer one.
#
# Each entry is (regex, offset_days, part_of_day). An offset of None means the
# handler computes it (weekdays, "in N days").
_DAY_PATTERNS: list[tuple[str, int | None, str | None]] = [
    # --- explicit two- and three-word phrases first
    (r"\bday after tomorrow\b", 2, None),
    (r"\bthe day after tomorrow\b", 2, None),
    (r"\btomorrow morning\b", 1, PART_MORNING),
    (r"\btomorrow afternoon\b", 1, PART_AFTERNOON),
    (r"\btomorrow evening\b", 1, PART_EVENING),
    (r"\btomorrow night\b", 1, PART_NIGHT),
    (r"\bthis morning\b", 0, PART_MORNING),
    (r"\bthis afternoon\b", 0, PART_AFTERNOON),
    (r"\bthis evening\b", 0, PART_EVENING),
    (r"\bthis night\b", 0, PART_NIGHT),
    # "tonight" is today plus a night flag, not a separate day: a user asking
    # at 18:00 about "tonight" means the hours still ahead of them today.
    (r"\btonight\b", 0, PART_NIGHT),
    (r"\bovernight\b", 0, PART_NIGHT),
    # --- single words
    (r"\btomorrow\b", 1, None),
    (r"\btomorow\b", 1, None),      # typo present in the intent training data
    (r"\btmrw\b", 1, None),
    (r"\btoday\b", 0, None),
    (r"\bright now\b", 0, None),
    (r"\bnow\b", 0, None),
    (r"\bat the moment\b", 0, None),
    (r"\bcurrently\b", 0, None),
    # --- "in the <part>", which means today's remaining hours.
    #     Bare "morning"/"evening" are deliberately NOT patterns: "good
    #     morning" and "have a good evening" are greetings, and the intent
    #     classifier already has `greet`/`goodbye` for them.
    (r"\bin the morning\b", 0, PART_MORNING),
    (r"\bin the afternoon\b", 0, PART_AFTERNOON),
    (r"\bin the evening\b", 0, PART_EVENING),
]

# "this weekend" -> Saturday. A weekend spans two days and the forecast is
# per-day, so one has to be chosen; Saturday is the first of the two and the
# day a tour is more likely to be planned for. The dialogue manager can offer
# Sunday as a follow-up.
_WEEKEND_RE = re.compile(r"\b(this |next |the )?weekend\b")

# "in 3 days", "in a week".
_IN_N_DAYS_RE = re.compile(r"\bin (\d+|a|one|two|three|four|five|six|seven) (day|days|week|weeks)\b")
_NUMBER_WORDS: dict[str, int] = {
    "a": 1, "one": 1, "two": 2, "three": 3,
    "four": 4, "five": 5, "six": 6, "seven": 7,
}

# "on Friday", "next Friday", "friday".
_WEEKDAY_RE = re.compile(
    r"\b(next |on |this |coming )?(" + "|".join(WEEKDAY_NAMES) + r")\b"
)


def today_in_vienna() -> date:
    """Return the current date in Austrian local time.

    Separate function so tests can monkeypatch "now" without touching the
    system clock.

    Returns:
        Today's date in ``Europe/Vienna``.
    """
    return datetime.now(VIENNA).date()


def _make_day(
    offset: int, part_of_day: str | None, matched_text: str, reference: date
) -> DaySlot:
    """Build a ``DaySlot`` and flag it if it is outside the forecast horizon.

    Args:
        offset: Days from the reference date.
        part_of_day: Part-of-day flag or ``None``.
        matched_text: The phrase that produced this.
        reference: The date that counts as "today".

    Returns:
        The day slot, with ``in_range=False`` if the bot cannot forecast it.
    """
    return DaySlot(
        target_date=reference + timedelta(days=offset),
        offset_days=offset,
        part_of_day=part_of_day,
        matched_text=matched_text,
        in_range=0 <= offset <= MAX_DAY_OFFSET,
    )


def find_day(message: str, reference: date | None = None) -> DaySlot:
    """Resolve the day a message refers to.

    Recognises ``today``, ``tonight``, ``this morning/afternoon/evening``,
    ``tomorrow``, ``the day after tomorrow``, the weekday names (with
    ``next``/``on``/``this``), ``this weekend`` (resolved to Saturday) and
    ``in N days``.

    Defaults to **today** when no day word is present, which matches what a
    user means by a bare "how windy is it on the Hochkönig".

    Dates beyond the forecast horizon are returned with ``in_range=False``
    rather than being clamped, so the bot can say it cannot forecast that far
    instead of answering for the wrong day.

    Args:
        message: Raw user message.
        reference: The date to treat as today. Defaults to today in Vienna;
            passed explicitly by the tests so they do not depend on the clock.

    Returns:
        The resolved day; never ``None``.
    """
    reference = reference or today_in_vienna()
    # Lower-case but keep word shapes; day words contain no umlauts, so the
    # full name normalisation is not needed here.
    text = " ".join(
        "".join(ch if ch.isalnum() else " " for ch in message.lower()).split()
    )

    # 1. Fixed phrases. Longest first by construction of the list.
    for pattern, offset, part in _DAY_PATTERNS:
        match = re.search(pattern, text)
        if match and offset is not None:
            return _make_day(offset, part, match.group(0), reference)

    # 2. "in N days" / "in a week"
    match = _IN_N_DAYS_RE.search(text)
    if match:
        raw_number, unit = match.group(1), match.group(2)
        count = int(raw_number) if raw_number.isdigit() else _NUMBER_WORDS[raw_number]
        offset = count * 7 if unit.startswith("week") else count
        return _make_day(offset, None, match.group(0), reference)

    # 3. "this weekend" -> the coming Saturday. Checked before the weekday
    #    patterns so that "next weekend" is not read as a bare weekday.
    match = _WEEKEND_RE.search(text)
    if match:
        prefix = (match.group(1) or "").strip()
        days_until_saturday = (5 - reference.weekday()) % 7
        # On a Saturday, "this weekend" means today; on a Sunday it has
        # already started, so the coming Saturday is six days out and the
        # modulo above gives that naturally.
        offset = days_until_saturday
        if prefix == "next":
            offset += 7
        return _make_day(offset, None, match.group(0), reference)

    # 4. Weekday names.
    match = _WEEKDAY_RE.search(text)
    if match:
        prefix = (match.group(1) or "").strip()
        weekday = WEEKDAY_NAMES[match.group(2)]
        days_until = (weekday - reference.weekday()) % 7
        # "Friday" said on a Friday means today, not a week from now.
        offset = days_until
        if prefix == "next":
            # "next Friday" is read as the Friday of the following week. This
            # is genuinely ambiguous in English; the convention is documented
            # here and in the tests so the behaviour is at least predictable,
            # and it will usually fall outside the 7-day horizon, where the
            # bot says so rather than guessing.
            offset = days_until + 7
        return _make_day(offset, None, match.group(0), reference)

    # 5. Nothing found: assume today.
    return DaySlot(
        target_date=reference,
        offset_days=0,
        part_of_day=None,
        matched_text="",
        is_default=True,
        in_range=True,
    )


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
        The place (or ``None``) and the resolved day.
    """
    return Slots(
        place=find_place(message, csv_path=csv_path),
        day=find_day(message, reference=reference),
    )
