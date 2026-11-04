"""Rule-based safety guard: avalanche questions never reach the classifier.

Why this module exists
----------------------
PeakCast informs about weather. It explicitly does **not** replace official
mountain weather or avalanche reports (see the concept and README). Avalanche
questions are therefore the one error class with a real-world safety
consequence: answering "what is the avalanche risk on the Dachstein tomorrow?"
with a snowfall forecast would read as reassurance the bot is not entitled to
give.

Two measurements from the earlier steps show that the intent classifier cannot
be trusted with this on its own (DEVLOG.md, steps 3 and 4):

    "avalanche risk after the fresh snow"
        baseline      -> ask_precipitation, p = 0.25
        iteration 1   -> ask_precipitation, p = 0.62

The *better* model is *more* confidently wrong, and p = 0.62 sits far above the
0.35 fallback threshold, so the confidence fallback does not catch it either.
The cause is structural rather than fixable with more data: "fresh snow" is
strong precipitation evidence, and a bag-of-features model has no concept of
what the question is *for*.

Design decisions
----------------
* **Order: guard first, classifier second.** A guard running *after* the
  classifier would only ever see the messages the model already mislabelled,
  which is exactly the set it cannot be relied on to produce.
* **Matching is deliberately tolerant** (substring on a normalised message, plus
  a short list of misspellings), because a false *positive* here is cheap - the
  user is pointed at the official avalanche service, which is good advice for
  anyone on a winter mountain - while a false *negative* is the one failure the
  project must not have. The asymmetry justifies erring towards catching.
* **No forecast values are attached** to the referral, not even the ones the
  bot could legitimately produce. Mixing "here is 20 cm of fresh snow" into a
  safety referral would undo the point of referring.
* The avalanche sentences **stay in the `out_of_scope` training data** as well.
  Guard and training data are two independent layers: if a phrasing ever slips
  past the pattern, the classifier should still not route it to a weather
  answer.

The guard is intentionally a plain function with no model and no state, so its
behaviour is fully auditable - which is the property that makes it suitable for
a safety requirement in the first place.
"""

from __future__ import annotations

import re
import unicodedata

# --- The referral answer ---------------------------------------------------
# One fixed string, with no forecast values interpolated into it.
#
# Which service to name was corrected during the step 6 review, and the
# distinction matters for a safety referral:
#
#   lawinen.at        the joint portal of the Austrian avalanche warning
#                     services, with a map covering ALL federal states. In
#                     Austria avalanche warning is organised per state, so this
#                     is the correct nationwide entry point.
#   avalanche.report  covers only Tyrol, South Tyrol and Trentino (the
#                     Euregio service). Naming it as if it were nationwide
#                     would send someone in Styria or Carinthia to a bulletin
#                     that does not cover their mountain.
#
# So the default referral points at lawinen.at, and avalanche.report is added
# only when the question is about a place in Tyrol, where it genuinely applies.
AVALANCHE_REFERRAL: str = (
    "I can't answer avalanche questions - and I shouldn't, because getting that "
    "wrong is dangerous. Avalanche warning in Austria is run per federal state; "
    "the joint portal with a map for the whole country is https://lawinen.at. "
    "I can still tell you the temperature, sky, wind or precipitation forecast "
    "for a place if that helps your planning."
)

# Added to the referral when the message named a place in Tyrol.
TYROL_REFERRAL_SUFFIX: str = (
    " For Tyrol you can also use https://avalanche.report, the joint bulletin "
    "for Tyrol, South Tyrol and Trentino."
)

# The federal state name as it appears in ``peakcast.slots.Place.state``
# (and in data/places_at.csv), kept as a named constant so the guard does not
# depend on a string literal matching the gazetteer by luck.
TYROL_STATE: str = "Tirol"

# --- Trigger patterns ------------------------------------------------------
# Matched as substrings against the normalised message (lower-cased, accents
# stripped, punctuation turned into spaces). Substring matching rather than
# whole-word matching is what makes German compounds work: "Lawinengefahr",
# "Lawinenlage" and "Lawinenwarnstufe" all contain "lawine" only as a prefix
# "lawinen", so both stems are listed.
#
# The English misspellings are the ones a hurried user actually produces: a
# dropped final "e", a swapped "ch"/"c", a missing "l".
AVALANCHE_PATTERNS: tuple[str, ...] = (
    # --- English, correct and misspelled
    "avalanche",
    "avalanch",    # dropped final "e"
    "avalance",    # "ch" -> "c"
    "avlanche",    # dropped "a"
    "avalanshe",
    "avelanche",   # "a" -> "e"
    "avalaunch",
    "snowslide",
    "snow slide",
    # --- German, correct and misspelled. "lawin" covers Lawine, Lawinen-,
    #     Lawinengefahr, Lawinenlage, Lawinenwarndienst, Lawinenwarnstufe,
    #     lawinengefährlich, and the common typo "Lawiene".
    "lawin",
    "lawien",      # transposed "ie"
    "schneebrett",  # slab avalanche - the technical term a ski tourer uses
)

# Matched as whole words only. "lwd" (Lawinenwarndienst) is three letters and
# would otherwise fire inside unrelated strings; a word boundary keeps it safe.
# Word boundaries are applied to the normalised message, where punctuation has
# already become whitespace, so "LWD-Bericht?" still matches.
AVALANCHE_WORD_PATTERNS: tuple[str, ...] = (
    "lwd",          # Lawinenwarndienst, the common Austrian abbreviation
    "slab",         # "slab conditions", "wind slab" - avalanche terminology
)

# Pre-compiled whole-word regex for the short patterns above.
_WORD_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in AVALANCHE_WORD_PATTERNS) + r")\b"
)


def normalise(message: str) -> str:
    """Lower-case a message, strip accents and turn punctuation into spaces.

    Normalising before matching is what lets one pattern cover several surface
    forms: ``"Avalanche-Risk?"``, ``"AVALANCHE RISK"`` and
    ``"Lawinengefahr"`` / ``"Lawinengefaehr"`` all reduce to something the
    plain lower-case patterns find.

    Accents are removed via NFKD decomposition, so "ä" becomes "a" rather than
    being dropped. Note that this maps "ö"->"o" and "ü"->"u" but **not**
    "ö"->"oe", so the ASCII spellings users type on a non-German keyboard are
    handled separately where that matters (see ``peakcast.slots``).

    Args:
        message: Raw user message.

    Returns:
        The normalised message, with single spaces between tokens.
    """
    decomposed = unicodedata.normalize("NFKD", message.lower())
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    # Keep letters, digits and spaces; everything else becomes a separator so
    # that "avalanche-risk" and "lawine?" still expose their stems.
    cleaned = "".join(ch if ch.isalnum() else " " for ch in without_accents)
    return " ".join(cleaned.split())


def mentions_avalanche(message: str) -> bool:
    """Report whether a message asks about avalanches.

    Args:
        message: Raw user message.

    Returns:
        ``True`` if any avalanche pattern matches.
    """
    normalised = normalise(message)
    if any(pattern in normalised for pattern in AVALANCHE_PATTERNS):
        return True
    return bool(_WORD_RE.search(normalised))


def referral_for(state: str | None = None) -> str:
    """Build the referral text, adding the Tyrol bulletin where it applies.

    Args:
        state: Federal state of the place the user asked about, if one was
            extracted. ``None`` when no place was named.

    Returns:
        The referral string.
    """
    if state == TYROL_STATE:
        return AVALANCHE_REFERRAL + TYROL_REFERRAL_SUFFIX
    return AVALANCHE_REFERRAL


def check_message(message: str, state: str | None = None) -> str | None:
    """Screen a message before it reaches the intent classifier.

    This is the function the dialogue manager calls **first**, on every turn.

    Note on ordering: the guard itself does not need a place to do its job, so
    it never waits for one. The ``state`` argument only *refines* the referral,
    which is why it is optional - if place extraction fails or is skipped, the
    user still gets the correct nationwide portal.

    Args:
        message: Raw user message.
        state: Federal state of the place named in the message, if the slot
            extractor found one. Only used to add the Tyrol-specific bulletin.

    Returns:
        The referral string if the message must not be answered with a
        forecast, otherwise ``None`` to mean "safe to classify normally".
    """
    if mentions_avalanche(message):
        return referral_for(state)
    return None
