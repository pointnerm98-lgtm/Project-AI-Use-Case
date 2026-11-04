"""Unit tests for the avalanche safety guard.

These tests carry more weight than the rest of the suite. The guard is the
project's answer to the one failure with a real-world safety consequence
(DEVLOG.md, steps 2 and 4), so its behaviour has to be pinned down rather than
spot-checked - including the case that the intent classifier provably gets
wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from peakcast.safety_guard import (
    AVALANCHE_REFERRAL,
    check_message,
    mentions_avalanche,
    normalise,
)

# --- The case the classifier gets wrong ------------------------------------
# Measured in step 4: the iteration-1 model labels this ask_precipitation with
# p = 0.62, far above the 0.35 fallback threshold. The guard is the only layer
# that stops it, which is exactly why it gets its own named test.
CLASSIFIER_FAILURE_CASE = "avalanche risk after the fresh snow"


def test_the_sentence_the_classifier_answers_wrongly_is_caught() -> None:
    """The step 4 counter-example must be intercepted before classification."""
    assert check_message(CLASSIFIER_FAILURE_CASE) == AVALANCHE_REFERRAL


def test_referral_contains_no_forecast_and_names_the_official_service() -> None:
    """The referral must point at the official service and promise no values."""
    assert "avalanche.report" in AVALANCHE_REFERRAL
    assert "Lawinenwarndienst" in AVALANCHE_REFERRAL
    # A safety referral that also reports snow depth would undo its own point.
    for number_word in ("cm", "km/h", "degrees", "°C"):
        assert number_word not in AVALANCHE_REFERRAL


@pytest.mark.parametrize(
    "message",
    [
        # --- plain English
        "what is the avalanche risk on the Dachstein",
        "whats the avalanche bulletin for Tirol",
        "Avalanche danger tomorrow?",
        "AVALANCHE RISK ON THE GLOCKNER",
        "avalanche-risk at the Kitzsteinhorn",
        # --- misspellings a hurried user produces
        "avalanch risk tomorrow",
        "avalance danger on the Rax",
        "avlanche conditions",
        "avelanche warning",
        # --- German, which an Austrian user will type
        "wie ist die Lawinengefahr am Dachstein",
        "Lawine am Hochkoenig?",
        "lawinenwarnstufe heute",
        "Lawinenlage Tirol",
        # --- German with ASCII umlauts and without
        "lawinengefaehrlich morgen",
        # --- abbreviations and technical terms
        "LWD Bericht fuer Tirol",
        "whats the LWD-Bericht?",
        "is there a wind slab on the north face",
        "schneebrett danger",
        "snow slide risk",
    ],
)
def test_avalanche_phrasings_are_caught(message: str) -> None:
    """Every avalanche phrasing must trigger the guard."""
    assert mentions_avalanche(message), f"guard missed: {message!r}"
    assert check_message(message) == AVALANCHE_REFERRAL


@pytest.mark.parametrize(
    "message",
    [
        # --- the ordinary weather questions the bot exists to answer
        "how windy is it on the Hochkoenig tomorrow",
        "will it snow in Soelden tonight",
        "how much fresh snow on the Kitzsteinhorn",
        "is there powder on the Dachstein",
        "temperature in Lienz on Saturday",
        "hows the weather in Zell am See",
        "do I need a rain jacket in Salzburg today",
        "will the trail be wet on the Rax tomorrow",
        "are the lifts running at the Turracher Hoehe",
        "hello",
        "thanks bye",
        # --- snow words on their own must not trigger it: the guard is about
        #     avalanches, and over-triggering on "snow" would break the
        #     ask_precipitation intent entirely
        "fresh snow tomorrow please",
        "snow depth on the Grossglockner",
    ],
)
def test_normal_weather_questions_pass_through(message: str) -> None:
    """Non-avalanche messages must reach the classifier untouched."""
    assert not mentions_avalanche(message), f"false positive: {message!r}"
    assert check_message(message) is None


def test_normalisation_folds_case_accents_and_punctuation() -> None:
    """Normalisation is what lets one pattern cover many surface forms."""
    assert normalise("Avalanche-Risk?") == "avalanche risk"
    assert normalise("LAWINENGEFAHR!!") == "lawinengefahr"
    assert normalise("Lawinengefähr") == "lawinengefahr"
    assert normalise("  multiple   spaces  ") == "multiple spaces"


def test_empty_message_is_not_an_avalanche_question() -> None:
    """An empty or whitespace-only message must not trigger the guard."""
    assert check_message("") is None
    assert check_message("   ") is None


def test_guard_is_deterministic() -> None:
    """The same message must always give the same verdict.

    Trivial for a rule, and that is the point: this property is what makes the
    guard auditable in a way the classifier is not.
    """
    for _ in range(3):
        assert check_message(CLASSIFIER_FAILURE_CASE) == AVALANCHE_REFERRAL
        assert check_message("how windy is it tomorrow") is None
