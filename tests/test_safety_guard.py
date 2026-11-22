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
    TYROL_REFERRAL_SUFFIX,
    check_message,
    mentions_avalanche,
    normalise,
    referral_for,
)
from peakcast.slots import find_place

# --- The case the classifier gets wrong ------------------------------------
# Measured in step 4: the iteration-1 model labels this ask_precipitation with
# p = 0.62, far above the 0.35 fallback threshold. The guard is the only layer
# that stops it, which is exactly why it gets its own named test.
CLASSIFIER_FAILURE_CASE = "avalanche risk after the fresh snow"


def test_the_sentence_the_classifier_answers_wrongly_is_caught() -> None:
    """The step 4 counter-example must be intercepted before classification."""
    assert check_message(CLASSIFIER_FAILURE_CASE) == AVALANCHE_REFERRAL


def test_referral_names_the_nationwide_portal() -> None:
    """The default referral must point at the all-Austria portal.

    lawinen.at is the joint portal of the state avalanche warning services and
    covers every federal state. avalanche.report covers only Tyrol, South Tyrol
    and Trentino, so it must NOT appear in the default referral - naming it
    would send someone in Styria or Carinthia to a bulletin that does not cover
    their mountain.
    """
    assert "lawinen.at" in AVALANCHE_REFERRAL
    assert "avalanche.report" not in AVALANCHE_REFERRAL


def test_referral_contains_no_forecast_values() -> None:
    """A safety referral that also reports snow depth would undo its own point."""
    for unit in ("cm", "km/h", "degrees", "°C"):
        assert unit not in AVALANCHE_REFERRAL
        assert unit not in TYROL_REFERRAL_SUFFIX


def test_tyrol_gets_the_euregio_bulletin_as_well() -> None:
    """In Tyrol, avalanche.report genuinely applies and is added."""
    referral = referral_for("Tirol")
    assert "lawinen.at" in referral
    assert "avalanche.report" in referral


@pytest.mark.parametrize("state", ["Steiermark", "Kärnten", "Salzburg", "", None])
def test_other_states_do_not_get_the_tyrol_bulletin(state: str | None) -> None:
    """Outside Tyrol the Euregio bulletin must not be offered."""
    referral = referral_for(state)
    assert "lawinen.at" in referral
    assert "avalanche.report" not in referral


def test_referral_is_refined_by_the_extracted_place() -> None:
    """End to end: a Tyrolean place in the message adds the Tyrol bulletin.

    Uses the real slot extractor rather than a hand-written state string, so
    the test also pins the contract between the two modules - the guard reads
    the state exactly as `Place.state` spells it.
    """
    tyrolean = find_place("what is the avalanche risk on the Wildspitze")
    assert tyrolean is not None and tyrolean.state == "Tirol"
    assert "avalanche.report" in check_message(
        "what is the avalanche risk on the Wildspitze", tyrolean.state
    )

    styrian = find_place("avalanche danger on the Hochschwab")
    assert styrian is not None and styrian.state == "Steiermark"
    assert "avalanche.report" not in check_message(
        "avalanche danger on the Hochschwab", styrian.state
    )


def test_guard_works_without_a_place() -> None:
    """The guard must never depend on place extraction succeeding.

    A bare "avalanche risk?" has no place in it, and the user must still get
    the correct nationwide portal.
    """
    assert check_message("avalanche risk?") == AVALANCHE_REFERRAL


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


# --- Step 11: false positives of the fuzzy layer, measured on project data ----
# The fuzzy layer added after user test round 1 is only acceptable if it does
# not start catching ordinary messages. These tests run the guard over every
# sentence and every place name the project has and pin the exact set that is
# caught. A new false positive fails them, and so does a lost true positive.
import csv as _csv  # noqa: E402

_DATA = Path(__file__).resolve().parent.parent / "data"

# Every sentence in the intent and slot data that the guard catches - each one
# checked by hand to really be about avalanches. (They are the avalanche
# phrasings kept in out_of_scope on purpose; see the module docstring.)
EXPECTED_CAUGHT_DATA_SENTENCES = {
    "what is the avalanche risk on the Dachstein",       # intents_train
    "whats the avalanche bulletin for Tirol",            # intents_train
    "whats the avalanche danger level in Vorarlberg",    # intents_test
}

# The one gazetteer name the guard catches: a mountain whose name literally
# contains "Lawine" (Totes Gebirge). It was caught by the substring stems before
# step 11 too; the asymmetry argument in the module docstring accepts it.
EXPECTED_CAUGHT_PLACE_NAMES = {"Lawinenstein"}


def _read_column(name: str, column: str) -> list[dict[str, str]]:
    with (_DATA / name).open(encoding="utf-8", newline="") as handle:
        return [row for row in _csv.DictReader(handle)]


def test_no_false_positives_on_intent_and_slot_data() -> None:
    """Over all intent sentences and the slot test set, only avalanche ones."""
    sentences: set[str] = set()
    for name, column in (
        ("intents_train.csv", "text"),
        ("intents_test.csv", "text"),
        ("intents_train_round2.csv", "text"),
        ("intents_test_round2.csv", "text"),
        ("slots_test.csv", "sentence"),
    ):
        sentences.update(row[column] for row in _read_column(name, column))
    caught = {s for s in sentences if mentions_avalanche(s)}
    assert caught == EXPECTED_CAUGHT_DATA_SENTENCES


def test_user_test_sentences_caught_iff_safety_rows() -> None:
    """In the user-test file, exactly the safety rows trigger the guard."""
    for row in _read_column("user_test_round1.csv", "user_message"):
        for message in (row["user_message"], row["previous_message"]):
            if not message:
                continue
            expected = row["area"] == "safety" and message == row["user_message"]
            assert mentions_avalanche(message) == expected, message


def test_no_false_positives_on_place_names() -> None:
    """No gazetteer name except the Lawinenstein triggers the guard.

    Measured during step 11: with two edits allowed on the short German word
    "lawine", 29 names were caught (Laming, Lading, Labing, Tafine, ...), which
    is why the German targets allow only one edit.
    """
    names: set[str] = set()
    for row in _read_column("places_at.csv", "name"):
        names.add(row["name"])
        names.update(a for a in row["alt_names"].split("|") if a)
    caught = {name for name in names if mentions_avalanche(name)}
    assert caught == EXPECTED_CAUGHT_PLACE_NAMES
